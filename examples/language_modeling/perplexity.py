import argparse
import logging

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantized_training import add_training_args, quantize_model, run_task

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process model parameters.")
    parser.add_argument('--model_id', type=str, required=True, help='Pretrained model identifier')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--stride', type=int, default=512, help='Stride for processing the data')
    add_training_args(parser)
    return parser.parse_args()

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16, # torch.float16 cause overflow
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    quantize_model(model, args)

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, args.stride)):
        end_loc = min(begin_loc + args.max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
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

if __name__ == "__main__":
    args = parse_args()
    run_task(args, main)