def Get_Model(args):
    if args.Model_name in ['BERT', 'RoBERTa', 'BGE', 'SAILER', 'Lawformer']:
        from .DualEncoder import DualEncoder
        return DualEncoder(args)
    elif args.Model_name == 'BERTPLI':
        from .BERTPLI import BERT_PLI
        return BERT_PLI(args)
    elif args.Model_name == 'DistillReasoning':
        from .DistillReasoning import DistillModel
        return DistillModel(args)
    elif args.Model_name == 'KELLER':
        from .KELLER import KELLER
        return KELLER(args)
    else:
        raise NotImplementedError