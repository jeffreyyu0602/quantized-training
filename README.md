# Quantized-Training

Quantized-Training is a Python package designed to facilitate the efficient quantization of Transformer networks and other Deep Neural Networks (DNNs), with a particular focus on Huggingface's Transformers implementation. It supports quantization using FP8 and Posit8 data types, aiming to enhance performance and efficiency in machine learning tasks, especially those involving large Transformer models.

## Features

- **Support for Custom Models:** Provides flexibility by allowing users to add their own model implementations for quantization.
- **Quantization Data Types:** Supports FP8 (E4M3 and E5M2) and posit with arbitrary nbits and es
- **Ease of Use:** Comes in a packaged format for simple installation and includes example usage to help users get started quickly.

## Prerequisites

- **Python Version:** Python 3.9 or newer.
- **PyTorch:** Version 2.0 or newer.

## Installation

Ensure you meet the prerequisites above. To install Quantized-Training directly from the source using pip, follow these steps:

```bash
git clone https://github.com/jeffreyyu0602/quantized-training.git
cd quantized-training
pip install -r requirements.txt
pip install -e .
```

## Usage

After installation, Quantized-Training can be easily used in your projects. Here's a quick start example:

```python
from quantized_training import add_training_args, quantize
```

Create an argument parser and add quantized training relevant arguments by calling add_training_args
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, required=True, help='Pretrained model identifier')
add_training_args(parser)
args= parser.parse_args()

```

Initialize your Transformer model from Huggingface's implementation
```python
from transformers import AutoModel
model = AutoModel.from_pretrained(model_id)
```

Initialize the quantizer for your model with the desired quantization type
```python
quantize(model, args)
```

Your model is now quantized and ready for training or inference

For users interested in adding support for their own models, please refer to the `example` folder for guidelines and examples on how to extend the functionality of Quantized-Training.

## Results and Reproduction

#### SQuAD Inference

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

## Contributing

We welcome contributions to quantized-Training! If you have suggestions for improvements or new features, please feel free to contribute. Check out the CONTRIBUTING.md file for guidelines on how to submit contributions.

## License

Quantized-Training is released under the MIT License. See the LICENSE file for more details.

## Contact

If you have any questions or feedback regarding Quantized-Training, please contact [Jeffrey Yu](jeffreyy@stanford.edu)