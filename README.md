# 8-bit Transformer Inference and Fine-Tuning for Edge Accelerators
[[Paper](https://dl.acm.org/doi/10.1145/3620666.3651368)][[Slides](https://drive.google.com/file/d/16v3UhnWab2K_1wiDTYgXy1ZUNN-hCi7M/view?usp=sharing)][[Video](https://www.youtube.com/watch?v=lqW-8MQ2uFw)]

**Efficient and accurate** quantization for Transformers and CNNs, supporting **activation, weight, and gradient** quantization to **integer, floating-points, and posit** data types.

![overview](figures/overview.png)

The current release supports:

- **Quantization for Custom Models:** Provides flexibility by allowing users to add their own model implementations for quantization.
- **Quantization Data Types:** Supports integer with arbitrary bit width, FP8 (E4M3 and E5M2), FP6 (E3M2 and E2M3), FP4 (E2M1), and posit with customizable nbits and es.
- Examples on Google MobileBERT and RoBERTa fine-tuning on GLUE and SQuAD question answering task.

## News

## Content

## Prerequisites

- **Python Version:** Python 3.9 or newer.
- **PyTorch:** Version 2.3 or newer.

## Quickstart

1. Clone this repository and navigate to quantized-training folder
```bash
git clone https://github.com/jeffreyyu0602/quantized-training.git
cd quantized-training
```

2. Install Package
```bash
conda create -n qt python=3.9
conda activate qt
pip install -r requirements.txt
pip install -e .
```

3. Create an argument parser and add relevant quantization arguments by calling add_training_args.

```python
import argparse
from quantized_training import add_training_args

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, required=True, help='Pretrained model identifier')
add_training_args(parser)
args= parser.parse_args()

```

4. Initialize model and call quantizer on the model with parsed arguments.
```python
from transformers import AutoModel
from quantized_training import quantize

model = AutoModel.from_pretrained(model_id)

quantize(model, args)
```

Your model is now quantized and ready for training or inference. For more use cases, please refer to the `example` folder for guidelines and examples on how to extend the functionality of Quantized-Training.

## Quantizatiion Arguments



## Results on SQuAD Question Answering

To reproduce the Table 1 results in the paper, run
```python
python examples/question_answering/run_squad.py [--log_file <LOG_FILE>] [--out_file <OUTPUT>]
```
The outputs are stored in squad_f1.csv which has the same format as Table 1.

#### GLUE and SQuAD Fine-Tuning

Fine-tuning the Transformer models for sequence classification on the GLUE benchmark and question answering on the SQuAD v1.1. GLUE is made up of a total of 9 different tasks. In our paper, we conduct evaluations on three benchmarks: SST-2, MRPC, and QNLI. All commands required to reproduce the results presented in Table 4 are provided in the script named `asplos_training.sh`. The experiments are organized into groups, each addressing four tasks of different data types and configurations: BF16, Posit8, Posit8 with approximation, and FP8. Each task is repeated with three different random seeds to mitigate outlier results. Specifically, the first set of experiments involves running the MobileBERT-tiny model on the QNLI task across these four configurations. This setup corresponds to the results shown in the first major row (MobileBERT-tiny) and the first column (QNLI) of Table 4. The structure for subsequent groups of experiments follows the same pattern. Outputs from the experiments are recorded in their respective log files, as indicated by the log_file argument. We recommend starting with the MRPC task, as it is the shortest and typically completes in around an hour on an RTX 4090 GPU.

#### Whisper Evaluation

Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. To evaluate Whisper models on LibriSpeech test-clean:
```python
python examples/speech_recognition/librispeech_asr.py --model_id openai/whisper-tiny [--quantize_weight] [--quantize_fwd <OPERATIONS>]
```
where model_id could be any Whisper model in the [Whisper Release](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013).

The user can perform quantized inference by passing quantize_weight and quantize_fwd arguments. OPERATIONS could be any combination of "gemm", "act", "norm", "attn_scaling", and "residual", separated by comma.

#### LLM Evaluation

To run language models evaluation on WikiText-103:
```python
python examples/language_modeling/wikitext.py --model_id gpt2-xl [--max_length <LENGTH>] [--stride <STRIDE>]
```

To run LLaMA2, you need to first request access to models checkpoint on the [huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf) website. Then login in the terminal using [huggingface cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli). After the request has been granted, you will be able to run LLaMA2 with the script.

## Reference

If you find AWQ useful or relevant to your research, please kindly cite our paper:

```
@inproceedings{yu20248bit,
title = {8-bit Transformer Inference and Fine-tuning for Edge Accelerators},
author = {Yu, Jeffrey and Prabhu, Kartik and Urman, Yonatan and Radway, Robert M. and Han, Eric and Raina, Priyanka},
year = {2024},
url = {https://doi.org/10.1145/3620666.3651368},
doi = {10.1145/3620666.3651368},
booktitle = {Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3},
}
```
