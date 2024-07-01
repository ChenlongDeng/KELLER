from transformers import AutoModel, MT5EncoderModel, MT5ForConditionalGeneration, BertModel, BertLayer
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

def kernel_mu(n_kernels):
    """
    get mu for each guassian kernel, Mu is the middele of each bin
    :param n_kernels: number of kernels( including exact match). first one is the exact match
    :return: mus, a list of mu
    """
    mus = [1]  # exact match
    if n_kernels == 1:
        return mus
    bin_step = (1 - (-1)) / (n_kernels - 1)  # score from [-1, 1]
    mus.append(1 - bin_step / 2)  # the margain mu value
    for k in range(1, n_kernels - 1):
        mus.append(mus[k] - bin_step)
    return mus


def kernel_sigma(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels(including the exact match)
    :return: sigmas, a list of sigma
    """
    sigmas = [0.001]  # exact match small variance means exact match ?
    if n_kernels == 1:
        return sigmas
    return sigmas + [0.1] * (n_kernels - 1)

class KELLER(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        # Copy args from main function
        print('Initializing model...')
        self.Model_name = args.Model_name
        self.max_crime_num = args.max_crime_num
        self.negative_cross_device = args.negative_cross_device and dist.is_initialized()
        self.Encoder = AutoModel.from_pretrained(args.PLM_path)
        self.loss_fct = nn.CrossEntropyLoss()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        self.Encoder.gradient_checkpointing_enable()
    
    def _gather_tensors(self, local_tensor):
        """
        Gather tensors from all gpus on each process.

        Args:
            local_tensor: the tensor that needs to be gathered

        Returns:
            concatenation of local_tensor in each process
        """
        if local_tensor is None:
            return None
        all_tensors = [torch.empty_like(local_tensor)
                       for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, local_tensor.contiguous())
        all_tensors[dist.get_rank()] = local_tensor
        return torch.cat(all_tensors, dim=0)
    
    def forward(self, **kwargs):
        batch_size = int(kwargs['query_input_ids'].shape[0] / self.max_crime_num)
        doc_num_per_query = int(kwargs['doc_input_ids'].shape[0] / (batch_size * self.max_crime_num))
        query_input = {k: kwargs['query_'+k] for k in ['input_ids', 'attention_mask']}
        doc_input = {k: kwargs['doc_'+k].reshape(-1, kwargs['doc_'+k].shape[-1]) for k in ['input_ids', 'attention_mask']}
        query_reps = self.Encoder(**query_input)['last_hidden_state'][:, 0, :].reshape(batch_size, self.max_crime_num, -1) # (bs, seq, emb)
        query_reps = F.normalize(query_reps, p=2, dim=-1)
        doc_reps = self.Encoder(**doc_input)['last_hidden_state'][:, 0, :].reshape(batch_size, -1, self.max_crime_num, query_reps.shape[-1]) # (bs, N, seq, emb)
        doc_reps = F.normalize(doc_reps, p=2, dim=-1)
        
        kwargs['query_seq_mask'] = kwargs['query_seq_mask'].view(batch_size, self.max_crime_num)
        kwargs['doc_seq_mask'] = kwargs['doc_seq_mask'].view(batch_size, -1, self.max_crime_num)
        if self.training == True:
            # Training Mode
            if self.negative_cross_device:
                query_reps = self._gather_tensors(query_reps)
                doc_reps = self._gather_tensors(doc_reps)
                
            interaction_mask_matrix = torch.einsum('bs, Bnq -> bBnsq', kwargs['query_seq_mask'], kwargs['doc_seq_mask']).bool()
            scores = torch.einsum('bse, Bnqe -> bBnsq', query_reps, doc_reps)# (bs, bs, N, seq, seq)
            scaled_scores = scores / 0.01
            masked_scores = scaled_scores.masked_fill(~interaction_mask_matrix, -float('inf'))
            
            fine_grained_scores = masked_scores.permute(0, 3, 1, 2, 4).reshape(batch_size*self.max_crime_num, -1)
            fine_grained_scores = fine_grained_scores.masked_fill(~kwargs['fine_grained_mask'].bool(), -float('inf'))
            fine_grained_max_index = torch.zeros(batch_size*self.max_crime_num, dtype=torch.long, device=fine_grained_scores.device)
            for i in range(batch_size*self.max_crime_num):
                start_idx = (i//self.max_crime_num)*(doc_num_per_query*self.max_crime_num)
                end_idx = start_idx + self.max_crime_num
                if kwargs['fine_grained_label'][i] != -1:
                    kwargs['fine_grained_label'][i] += start_idx
                fine_grained_max_index[i] = start_idx + fine_grained_scores[i, start_idx:end_idx].max(0)[1]
            fine_grained_label = torch.where(kwargs['fine_grained_label'] == -1, fine_grained_max_index, kwargs['fine_grained_label'])
            fine_grained_loss = self.loss_fct(fine_grained_scores[kwargs['query_seq_mask'].view(-1).bool()], fine_grained_label[kwargs['query_seq_mask'].view(-1).bool()].long())
            
            # maxsim
            max_scores = torch.max(masked_scores, dim=-1)[0]
            final_scores = torch.where(max_scores == -float('inf'), torch.zeros_like(max_scores), max_scores).sum(dim=-1)
            final_scores = final_scores.reshape(batch_size, -1)
            masked_final_scores = final_scores.masked_fill(~kwargs['contrastive_mask'].bool(), -float('inf'))
            if final_scores.shape[-1] == batch_size:
                loss = self.loss_fct(masked_final_scores, torch.arange(final_scores.shape[0], device=scores.device))
            else:
                loss = self.loss_fct(masked_final_scores, torch.arange(0, query_reps.shape[0], query_reps.shape[0]/batch_size, dtype=torch.long, device=final_scores.device))
            return SequenceClassifierOutput(
                loss=fine_grained_loss
            )
        
        else:
            # Eval/Test Mode
            interaction_mask_matrix = torch.bmm(kwargs['query_seq_mask'].unsqueeze(2), kwargs['doc_seq_mask']).bool()
            scores = torch.bmm(query_reps, doc_reps.squeeze(1).transpose(1,2)) # (bs, seq, seq)
            
            masked_scores = scores.masked_fill(~interaction_mask_matrix, -float('inf'))
            max_scores = torch.max(masked_scores, dim=-1)[0]
            final_scores = torch.where(max_scores == -float('inf'), torch.zeros_like(max_scores), max_scores)
            final_scores = final_scores.sum(dim=-1)
            return {
                'loss': torch.tensor(0, dtype=torch.float, device=query_reps.device),
                'scores': final_scores
            }
