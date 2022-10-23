# Span NLI BERT for ContractNLI

Span NLI BERT is a baseline produced by stanford NLP for ContractNLI. Their github repi can be found here: https://github.com/stanfordnlp/contract-nli-bert
This repo consists of three branches, as detailed below:
1. main: with context-aware data preprocessing (adding section description to subsections)
2. stanfordnlp: the Span NLI BERT baseline by stanfordnlp
3. context-prompt: prompt tuning, and initializing prompts similar to the methods described in https://arxiv.org/abs/2111.02643

## Usage

### Dataset

Clone the repository to your desired directory.
Download the dataset from [the dataset repository](https://stanfordnlp.github.io/contract-nli/) and unzip JSON files to `./data/` directory (you may specify a custom dataset path by modifying a configuration file).

### Prerequisite

Set up a CUDA environment.
We used CUDA 11.1.1, cuDNN 8.0.5, NCCL 2.7.8-1 and GCC 7.4.0 in our experiments.

You need Python 3.8+ to run the codes.
We used Python 3.8.5 in our experiments.

Install requirements with Pip.

```bash
pip install -r requirements.txt
```

If you want to use A100 GPUs as we did in our experiments, you may need to install PyTorch manually (not reflected in `requirements.txt`).

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Running experiments

You can run an experiment by feeding `train.py` an experiment configuration YAML file and specifying an output directory.

```bash
python train.py ./data/conf_base.yml ./output
```
On single GPU:
```bash
CUDA_VISIBLE_DEVICES="0" python -m train.py ./data/conf_base.yml ./output
```
On multiple GPUs (might have multiprocessing issues):
```bash
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 train.py ./data/conf_base.yml ./output
```

## Reproducibility

We have tagged the implementations that we used for the experiments in our paper.
Please note that we have changed NLI label name since the experiments.
You may need to alter "Entailment" to "true", "Contradiction" to "false" and "NotMentioned" to "na" in the dataset JSON files, or apply commit `b0c4987` as a patch.

We carried out the experiments on [ABCI](abci.ai), a GPU cluster with a PBS-like job queue.
While it would not run in most users' environment, we provide our experiment procedure so that users can implement a similar procedure for their clusters.

```bash
# Generate configuration files in ./params
python gen_params.py data/param_tmpl.py 100 ./params

# Run tuning
./run_tuning.sh -s 1 -n 10 ${SECRET_GROUP_ID} ./params ./results

# Pick 3 models with the best validation macro NLI accuracies and report the average
python aggregate_results.py -n 3 -m macro_label_micro_doc.class.accuracy -o aggregated_metrics.txt ./results
```

## ContractNLI License

ContractNLI is released under Apache 2.0.
Please refer attached "[LICENSE](./LICENSE)" for the exact terms.

This implementation has partially been derived from Huggingface's implementation of SQuAD BERT.
Please refer the commit log for the full changes.

When you use Span NLI BERT in your work, please cite the paper:

```bibtex
@inproceedings{koreeda-manning-2021-contractnli,
    title = "ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts",
    author = "Koreeda, Yuta  and
      Manning, Christopher D.",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021",
    publisher = "Association for Computational Linguistics"
}
```
