import argparse
import logging
import os

import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from quantized_training import (
    add_qspec_args,
    quantize,
    plot_histogram,
    plot_layer_range,
    setup_logging,
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Perform whisper model inference on LibriSpeech dataset.")
    parser.add_argument("--model_id", default="openai/whisper-tiny", help="Model to perform evaluation.")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--output_dir", default=None, help="Output directory for scores.")
    add_qspec_args(parser)
    return parser.parse_args()


@setup_logging
def main(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")

    librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(args.model_id)

    quantize(model, args)

    def map_to_pred(batch):
        array = [audio["array"] for audio in batch["audio"]]
        sampling_rate = batch["audio"][0]["sampling_rate"]
        input_features = processor(array, sampling_rate=sampling_rate, return_tensors="pt").input_features
        batch["reference"] = [processor.tokenizer._normalize(text) for text in batch['text']]

        if args.bf16:
            input_features = input_features.bfloat16()

        with torch.no_grad():
            predicted_ids = model.generate(input_features.to(device))
        transcription = [processor.decode(pi) for pi in predicted_ids]
        batch["prediction"] = [processor.tokenizer._normalize(t) for t in transcription]
        return batch

    result = librispeech_test_clean.map(map_to_pred, batched=True, batch_size=args.batch_size)

    wer = load("wer")
    logger.info(100 * wer.compute(references=result["reference"], predictions=result["prediction"]))

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
            f.write('\n'.join(result["prediction"]) + '\n')
        with open(os.path.join(args.output_dir, "references.txt"), "w") as f:
            f.write('\n'.join(result["reference"]) + '\n')
        if args.record_histogram:
            plot_histogram(model, args.output_dir)
            plot_layer_range(model, args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)

