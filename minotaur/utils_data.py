import os
import pickle
import torch


def save_tensors_to_file(data, dir, filename):
    """Helper function to save data to a file."""
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, filename), 'wb') as file:
        pickle.dump(data, file)

def split_data(data):
    layers = ['query_layer', 'key_layer', 'value_layer', 'attention_scores', 'attention_probs', 'context_layer']
    layers = [f'mobilebert.encoder.layer.{i}.attention.self.{layer}' for i in range(24) for layer in layers]
    # Manually split activations into multiple heads
    for name in layers:
        if name in data:
            for j in range(4):
                data[f'{name}.{j}'] = data[name][:, j]

def save_activations(activations, batch, output_dir, split_head=True):
    if split_head:
        split_data(activations)

    if 'classifier' in activations:
        padding = torch.full((activations["classifier"].shape[0], 14), torch.finfo(torch.float).min)
        activations["classifier"] = torch.cat((activations["classifier"], padding), axis=1)
        activations["mobilebert.labels"] = torch.eye(16)[batch["labels"].cpu()].to(torch.float)

    activations["mobilebert.attention_mask"] = (1.0 - batch["attention_mask"].cpu()) * torch.finfo(torch.float).min
    save_tensors_to_file(activations, output_dir, 'activations.pkl')

def save_errors(errors, output_dir):
    split_data(errors)
    if 'classifier' in errors:
        padding = torch.zeros((errors["classifier"].shape[0], 14))
        errors["classifier"] = torch.cat((errors["classifier"], padding), axis=1)
    save_tensors_to_file(errors, output_dir, 'errors.pkl')

def save_weights_and_grads(model, output_dir):
    weights, grads = {}, {}
    for name, param in model.named_parameters():
        key = name.replace('base_model.model.', '').replace('base_layer', '')
        weight = param.data.detach().float().cpu()
        weights[key] = weight.T if weight.dim() == 2 else weight
        if param.grad is not None:
            grad = param.grad.detach().float().cpu()
            grads[key] = grad.T if grad.dim() == 2 else grad

    classifier_name = 'classifier.modules_to_save.default' if 'classifier' not in weights else 'classifier'
    if f"{classifier_name}.weight" in weights:
        weights["classifier.weight"] = torch.cat((weights[f"{classifier_name}.weight"].T, torch.zeros((14, 512))), dim=0)
        padding = torch.full((14,), float('-inf'), dtype=torch.float)
        weights["classifier.bias"] = torch.cat((weights[f"{classifier_name}.bias"], padding), dim=0)
    save_tensors_to_file(weights, output_dir, 'weights.pkl')

    if f"{classifier_name}.weight" in grads:
        grads["classifier.weight"] = torch.cat((grads[f"{classifier_name}.weight"].T, torch.zeros((14, 512))), dim=0)
        grads["classifier.bias"] = torch.cat((grads[f"{classifier_name}.bias"], torch.zeros(14)), dim=0)
    save_tensors_to_file(grads, output_dir, 'gradients.pkl')