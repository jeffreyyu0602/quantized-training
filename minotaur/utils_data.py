import os
import pickle
import torch
import torch.nn.functional as F


def save_tensors_to_file(data_dict, dir, filename):
    """Helper function to save data to a file."""
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, filename), 'wb') as file:
        pickle.dump(data_dict, file)

def split_data(data_dict):
    layers = ['query_layer', 'key_layer', 'value_layer', 'attention_scores', 'attention_probs', 'context_layer']
    layers = [f'mobilebert.encoder.layer.{i}.attention.self.{layer}' for i in range(24) for layer in layers]
    # Manually split activations into multiple heads
    for name in layers:
        if name in data_dict:
            head_size = data_dict[name].shape[1]
            for j in range(head_size):
                data_dict[f'{name}.{j}'] = data_dict[name][:, j]

def save_activations(activations, batch, output_dir, split_head=True):
    if split_head:
        split_data(activations)

    if 'classifier' in activations:
        activations["classifier"] = F.pad(activations["classifier"], (0, 14, 0, 0), value=torch.finfo(torch.float).min)

    activations["mobilebert.labels"] = torch.eye(16)[batch["labels"].cpu()].to(torch.float)
    activations["mobilebert.attention_mask"] = (1.0 - batch["attention_mask"].cpu()) * torch.finfo(torch.float).min
    save_tensors_to_file(activations, output_dir, 'activations.pkl')

def save_errors(errors, output_dir):
    split_data(errors)
    if 'classifier' in errors:
        errors["classifier"] = F.pad(errors["classifier"], (0, 14, 0, 0))
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

    classifier_name = 'classifier' if 'classifier' in weights else 'classifier.modules_to_save.default'
    if f"{classifier_name}.weight" in weights:
        weights["classifier.weight"] = F.pad(weights[f"{classifier_name}.weight"].T, (0, 0, 0, 14))
        weights["classifier.bias"] = F.pad(weights[f"{classifier_name}.bias"], (0, 14), value=float('-inf'))
    save_tensors_to_file(weights, output_dir, 'weights.pkl')

    if f"{classifier_name}.weight" in grads:
        grads["classifier.weight"] = F.pad(grads[f"{classifier_name}.weight"].T, (0, 0, 0, 14))
        grads["classifier.bias"] = F.pad(grads[f"{classifier_name}.bias"], (0, 14))
    save_tensors_to_file(grads, output_dir, 'gradients.pkl')
