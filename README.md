<div align="center">
<h1>Learning Interpretable Legal Case Retrieval via Knowledge-Guided Case Reformulation[<a href="https://aclanthology.org/2024.emnlp-main.73/">Paper</a>]</h1>
<img src="./imgs/KELLER.png" width="80%" class="center">
</div>

KELLER, a legal knowledge-guided case reformulation approach based on large language models (LLMs) for effective and interpretable legal case retrieval.

## Usage
```python
# Single GPU
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path ./config/KELLER.yaml
```

```python
# Multi GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 run.py --config_path ./config/KELLER.yaml
```
or simply run the shell script:
```shell
bash run.sh
```

## Data
Most of the data has been released in this repo, and others can be found (or processed) by:

- The `unique_candidates.json` of the LeCaRD dataset can been easily produced by merging docs according to their doc_ids.
- The `ori_candidates` dir is the original candidate dir of the LeCaRDv2 dataset.

## Citation
```
@inproceedings{deng-etal-2024-learning,
    title = "Learning Interpretable Legal Case Retrieval via Knowledge-Guided Case Reformulation",
    author = "Deng, Chenlong  and
      Mao, Kelong  and
      Dou, Zhicheng",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.73",
    pages = "1253--1265",
    abstract = "Legal case retrieval for sourcing similar cases is critical in upholding judicial fairness. Different from general web search, legal case retrieval involves processing lengthy, complex, and highly specialized legal documents. Existing methods in this domain often overlook the incorporation of legal expert knowledge, which is crucial for accurately understanding and modeling legal cases, leading to unsatisfactory retrieval performance. This paper introduces KELLER, a legal knowledge-guided case reformulation approach based on large language models (LLMs) for effective and interpretable legal case retrieval. By incorporating professional legal knowledge about crimes and law articles, we enable large language models to accurately reformulate the original legal case into concise sub-facts of crimes, which contain the essential information of the case. Extensive experiments on two legal case retrieval benchmarks demonstrate superior retrieval performance and robustness on complex legal case queries of KELLER over existing methods.",
}
```