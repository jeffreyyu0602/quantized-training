import argparse
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from evaluate import load

from quantized_training import add_training_args, quantize_model


parser = argparse.ArgumentParser(description="Perform whisper model inference on LibriSpeech dataset.")
parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for scores.")
parser.add_argument("--observe_histogram", action="store_true", help="Record the histogram of activation.")
add_training_args(parser)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
else:
    device = torch.device("cpu")

librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)

if args.bf16:
    model.bfloat16()

def run_fn(model):
    sample = librispeech_test_clean[0]["audio"]
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
    if args.bf16:
        input_features = input_features.bfloat16()
    with torch.no_grad():
        model.generate(input_features.to(device))

quantize_model(model, args, run_fn, device=device)

if args.observe_histogram:
    for name, module in model.named_modules():
        if isinstance(module, torch.ao.quantization.FakeQuantizeBase):
            module.histogram_observer_enabled[0] = 1

def map_to_pred(batch):
    input_features = torch.cat([
        processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features for audio in batch["audio"]
    ], dim=0)
    batch["reference"] = [processor.tokenizer._normalize(text) for text in batch['text']]

    if args.bf16:
        input_features = input_features.bfloat16()

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to(device))

    transcription = [processor.decode(ids) for ids in predicted_ids]
    batch["prediction"] = [processor.tokenizer._normalize(trans) for trans in transcription]

    return batch

result = librispeech_test_clean.map(map_to_pred, batched=True, batch_size=args.batch_size)

wer = load("wer")
print(100 * wer.compute(references=result["reference"], predictions=result["prediction"]))

if args.output_dir is not None:
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, f"prediction.txt"), "w") as f:
        f.write('\n'.join(result["prediction"]) + '\n')

    with open(os.path.join(args.output_dir, "reference.txt"), "w") as f:
        f.write('\n'.join(result["reference"]) + '\n')

    for name, module in model.named_modules():
        if isinstance(module, torch.ao.quantization.FakeQuantizeBase):
            module.plot_histograms(os.path.join(args.output_dir, f'{name}.png'))