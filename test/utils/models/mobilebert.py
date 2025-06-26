import torch

from itertools import islice
from torch.utils.data import DataLoader
from tqdm import tqdm

from quantized_training import (
    convert_pt2e,
    prepare_pt2e,
    transform,
    compile,
)

from transformers import default_data_collator

from .utils import get_transform_args, get_compile_args

def load_model(args):
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer
    if args.model_name_or_path is None:
            args.model_name_or_path = "google/mobilebert-uncased"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        attn_implementation="eager",
    ).eval()

    if args.bf16:
        model.bfloat16()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer

def quantize_and_dump_model(model, quantizer, calibration_data, vector_stages, args):
    calibration_dataloader = DataLoader(calibration_data, collate_fn=default_data_collator, batch_size=1)

    compile_args = get_compile_args(args)
    transform_args = get_transform_args(args, vector_stages)

    batch = next(iter(calibration_dataloader))
    input_ids = batch["input_ids"]
    input_shape = input_ids.size()

    embedding_output = model.mobilebert.embeddings(
            input_ids=input_ids,
            token_type_ids=batch["token_type_ids"]
        )

    extended_attention_mask = model.mobilebert.get_extended_attention_mask(batch["attention_mask"], input_shape)

    head_mask = model.mobilebert.get_head_mask(None, model.config.num_hidden_layers)

    example_args = (embedding_output, extended_attention_mask, head_mask)

    class MobileBertWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mobilebert = model.mobilebert
            self.classifier = model.classifier

        def forward(self, hidden_states, attention_mask, head_mask):
            for i, layer_module in enumerate(self.mobilebert.encoder.layer):
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                )
                hidden_states = layer_outputs[0]

                if args.remove_duplicate:
                    break

            first_token_tensor = hidden_states[:, 0]
            output = self.classifier(first_token_tensor)
            return output
        
    quantizer.set_module_name("classifier", None)

    gm = prepare_pt2e(MobileBertWrapper(), quantizer, example_args)

    for batch in tqdm(islice(calibration_dataloader, 10), desc="Calibrating MobileBERT"):
        embedding_output = model.mobilebert.embeddings(
                input_ids=batch["input_ids"],
                token_type_ids=batch["token_type_ids"]
            )
        gm(embedding_output, extended_attention_mask, head_mask)
    
    convert_pt2e(gm, args.bias)

    old_output = gm(*example_args)

    transform(gm, example_args, **transform_args)

    gm.graph.print_tabular()
    new_output = gm(*example_args)

    compile(gm, example_args, **compile_args)
    return gm, old_output, new_output

def evaluate(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(dataset, collate_fn=default_data_collator, batch_size=1)

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating MobileBERT"):
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            correct_predictions += (prediction == inputs["labels"]).sum().item()
            total_samples += inputs["labels"].size(0)
    
    print(f"MobileBERT Accuracy: {correct_predictions / total_samples:.4f}")

def evaluate_gm(gm, dataset):
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data_label_pair in tqdm(dataset, desc="Evaluating Quantized MobileBERT"):
            label = data_label_pair["labels"]
            outputs = gm(
                data_label_pair["embedding_output"],
                data_label_pair["attention_mask"],
                data_label_pair["head_mask"],
            )
            
            logits = outputs
            prediction = torch.argmax(logits, dim=-1)
            if prediction == label:
                correct_predictions += 1
            total_samples += 1
    
    print(f"Quantized MobileBERT Accuracy: {correct_predictions / total_samples:.4f}")
    
    