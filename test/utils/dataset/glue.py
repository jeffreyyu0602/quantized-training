import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertPreTrainedModel,
    MobileBertPreTrainedModel,
    default_data_collator,
)

from .utils import write_tensor_to_file

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def retrieve_dataset(model, tokenizer, args):
    raw_datasets = load_dataset("glue", args.task_name)

    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding="max_length", max_length=128, truncation=True
        )
        result["labels"] = examples["label"]
        return result
    
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    eval_dataset = processed_datasets["validation"]
    train_dataset = processed_datasets["train"]
    return eval_dataset, train_dataset

def dump_dataset(output_dir, dataset, model):
    eval_dataloader = DataLoader(
        dataset, collate_fn=default_data_collator, batch_size=1
    )

    if isinstance(model, MobileBertPreTrainedModel):
        embeddings = model.mobilebert.embeddings
        head_mask = model.mobilebert.get_head_mask(None, model.config.num_hidden_layers)
    elif isinstance(model, BertPreTrainedModel):
        embeddings = model.bert.embeddings
        head_mask = model.bert.get_head_mask(None, model.config.num_hidden_layers)
    else:
        raise ValueError("Model not supported")
    
    preprocessed_dataset = []

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Dumping dataset")):
        embedding_output = embeddings(
            input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"]
        )
        attention_mask = (1.0 - batch["attention_mask"]) * torch.finfo(torch.float).min
        label = int(batch["labels"].item())

        folder = os.path.join(output_dir, f"{step}_{label}")
        os.makedirs(folder, exist_ok=True)

        preprocessed_dataset.append({
            "embedding_output": embedding_output,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "labels": batch["labels"]
        })
        write_tensor_to_file(
            embedding_output, os.path.join(folder, "hidden_states.bin")
        )
        write_tensor_to_file(attention_mask, os.path.join(folder, "attention_mask.bin"))
    return preprocessed_dataset
    