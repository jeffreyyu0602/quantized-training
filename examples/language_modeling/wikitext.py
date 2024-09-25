import argparse
import logging
import os

import torch
from accelerate.utils import get_max_memory
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantized_training import (
    add_qspec_args,
    get_default_quantizer,
    prepare_pt2e,
    quantize,
    plot_histogram,
    plot_layer_range,
    setup_logging,
)
from quantized_training.pt2e_utils import dispatch_model, get_device_map
from quantized_training.quantize_pt2e import convert_pt2e

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Process model parameters.")
    parser.add_argument('--model_id', required=True, help='Pretrained model identifier')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--stride', type=int, default=512, help='Stride for processing the data')
    parser.add_argument('--output_dir', default=None, help='Output directory for histograms')
    parser.add_argument(
        '--torch_dtype',
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help=(
            "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
            "dtype will be automatically derived from the model's weights."
        )
    )
    add_qspec_args(parser)
    return parser.parse_args()


@setup_logging
def main(args):
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        # device_map=args.gpu if args.gpu is not None else "auto",
        attn_implementation="eager", # flash attention is not supported
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # quantize(model, args)

    quantizer = get_default_quantizer(
        input_activation=args.activation,
        weight=args.weight,
        record_histogram=args.record_histogram,
        force_scale_power_of_two=args.force_scale_power_of_two,
        outlier_threshold=args.outlier_threshold,
    )

    input_ids = torch.randint(0, 100, (1, args.max_length), device=device)
    example_args = (input_ids,)
    example_kwargs = {"labels": input_ids.clone()}
    seq_len = torch.export.Dim("seq_length", min=3, max=args.max_length)
    dynamic_shapes = {"input_ids": {1: seq_len}, "labels": {1: seq_len}}
    model = prepare_pt2e(model, quantizer, example_args, example_kwargs, dynamic_shapes)

    max_memory = get_max_memory()
    if args.gpu is not None:
        assert args.gpu in max_memory
        max_memory = {args.gpu: max_memory[args.gpu]}
    else:
        # Reserve 2GB for activation
        reserved = 4 * 1024 ** 3
        max_memory = {
            k: v - reserved
            for k, v in max_memory.items() if isinstance(k, int) and v > reserved
        }
        max_memory = dict(sorted(max_memory.items(), key=lambda item: item[1], reverse=True))
    device_map = get_device_map(model, max_memory=max_memory)
    dispatch_model(model, (input_ids, example_kwargs["labels"]), device_map=device_map)

    def calibrate(model):
        validation = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        encodings = tokenizer("\n\n".join(validation["text"]), return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        for i, begin_loc in enumerate(tqdm(range(0, seq_len - args.max_length, args.stride))):
            end_loc = min(begin_loc + args.max_length, seq_len)
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            with torch.no_grad():
                model(input_ids, labels=target_ids)

            if i == args.calibration_steps - 1:
                break

    if args.calibration_steps > 0:
        calibrate(model)
        for module in model.modules():
            if isinstance(module, torch.ao.quantization.FakeQuantizeBase):
                module.disable_observer()

    if args.convert_model:
        model = convert_pt2e(model)

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    # Subtract max_length from seq_len to ensure that the last window has length max_length
    for begin_loc in tqdm(range(0, seq_len - args.max_length, args.stride)):
        end_loc = min(begin_loc + args.max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())

    logger.info(f"model:      {args.model_id}")
    logger.info(f"max length: {args.max_length}")
    logger.info(f"stride:     {args.stride}")
    logger.info(f"perplexity: {ppl.item()}")

    if args.record_histogram and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_histogram(model, args.output_dir)
        plot_layer_range(model, args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
