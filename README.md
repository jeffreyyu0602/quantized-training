# Quantized-Training

Quantized-Training is a Python package designed to facilitate the efficient quantization of Transformer networks and other Deep Neural Networks (DNNs), with a particular focus on Huggingface's Transformers implementation. It supports quantization using FP8 and Posit8 data types, aiming to enhance performance and efficiency in machine learning tasks, especially those involving large Transformer models.

## Features

- **Support for Custom Models:** Provides flexibility by allowing users to add their own model implementations for quantization, broadening its applicability.
- **Quantization Data Types:** Supports FP8 and Posit8, offering choices in quantization precision to balance between performance and accuracy.
- **Ease of Use:** Comes in a packaged format for simple installation and includes example usage to help users get started quickly.

## Prerequisites

- **Python Version:** Python 3.9 or newer.
- **PyTorch:** Version 2.0 or greater.

## Installation

Ensure you meet the prerequisites above. To install Quantized-Training directly from the source using pip, follow these steps:

```bash
git clone https://github.com/yourusername/quantized-training.git
cd quantized-training
pip install -e .
```

## Usage

After installation, Quantized-Training can be easily used in your projects. Here's a quick start example:

```python
from quantized_training import quantize_model
```

Initialize your Transformer model from Huggingface's implementation
```python
from transformers import AutoModel
model = AutoModel.from_pretrained('bert-base-uncased')
```

Initialize the quantizer for your model with the desired quantization type
```python
quantized_model = quantize_model(model, args)
```

Your model is now quantized and ready for training or inference
css
Copy code

For users interested in adding support for their own models, please refer to the `example` folder for guidelines and examples on how to extend the functionality of Quantized-Training.

## Results and Reproduction

##### SQuAD Inference

##### GLUE and SQuAD Fine-Tuning

##### LLaMA2
To run LLaMA2, you need to first request access to models checkpoint on the [huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf) website. Then login in the terminal using [huggingface cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli). After the request has been granted, you will be able to run LLaMA2 with the script.

## Contributing

We welcome contributions to quantized-Training! If you have suggestions for improvements or new features, please feel free to contribute. Check out the CONTRIBUTING.md file for guidelines on how to submit contributions.

## License

Quantized-Training is released under the [LICENSE NAME] License. See the LICENSE file for more details.

## Contact

If you have any questions or feedback regarding Quantized-Training, please contact [Jeffrey Yu](jeffreyy@stanford.edu)