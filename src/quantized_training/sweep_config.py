sweep_configurations = {}

# ========== MobileBERT Sweep Configurations ==========

sweep_configurations["mobilebert_mnli_lora_bf16"] = {
    "method": "grid",
    "name": "mnli_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"value": 30},
        "learning_rate": {"max": 14, "min": 5}
    }
}

sweep_configurations["mobilebert_qnli_lora_bf16"] = {
    "method": "grid",
    "name": "qnli_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"value": 50},
        "learning_rate": {"max": 14, "min": 5}
    }
}

sweep_configurations["mobilebert_mrpc_lora_bf16"] = {
    "method": "grid",
    "name": "mrpc_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"value": 30},
        "learning_rate": {"max": 14, "min": 5}
    }
}

sweep_configurations["mobilebert_sst2_lora_bf16"] = {
    "method": "grid",
    "name": "sst2_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"value": 60},
        "learning_rate": {"max": 14, "min": 5}
    }
}

sweep_configurations["mobilebert_squad_lora_bf16"] = {
    "method": "grid",
    "name": "squad_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "f1"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"value": 30},
        "learning_rate": {"max": 14, "min": 5}
    }
}

# ========== RoBERTa Base Sweep Configurations ==========

sweep_configurations["roberta_base_mnli_lora_bf16"] = {
    "method": "grid",
    "name": "mnli_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"value": 30},
        "learning_rate": {"max": 9, "min": 5}
    }
}

sweep_configurations["roberta_base_qnli_lora_bf16"] = {
    "method": "grid",
    "name": "qnli_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 32},
        "num_train_epochs": {"value": 25},
        "learning_rate": {"max": 8, "min": 4}
    }
}

sweep_configurations["roberta_base_mrpc_lora_bf16"] = {
    "method": "grid",
    "name": "mrpc_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"value": 30},
        "learning_rate": {"max": 9, "min": 5}
    }
}

sweep_configurations["roberta_base_mrpc_lora_bf16"] = {
    "method": "grid",
    "name": "sst2_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"value": 60},
        "learning_rate": {"max": 9, "min": 5}
    }
}

sweep_configurations["roberta_base_squad_lora_bf16"] = {
    "method": "grid",
    "name": "squad_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "f1"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"value": 30},
        "learning_rate": {"max": 9, "min": 5}
    }
}

# ========== RoBERTa Large Sweep Configurations ==========

sweep_configurations["roberta_large_mnli_lora_bf16"] = {
    "method": "grid",
    "name": "mnli_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 4},
        "num_train_epochs": {"value": 10},
        "learning_rate": {"max": 7, "min": 3}
    }
}

sweep_configurations["roberta_large_qnli_lora_bf16"] = {
    "method": "grid",
    "name": "qnli_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 4},
        "num_train_epochs": {"value": 10},
        "learning_rate": {"max": 6, "min": 2}
    }
}

sweep_configurations["roberta_large_mrpc_lora_bf16"] = {
    "method": "grid",
    "name": "mrpc_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 4},
        "num_train_epochs": {"value": 20},
        "learning_rate": {"max": 7, "min": 3}
    }
}

sweep_configurations["roberta_large_sst2_lora_bf16"] = {
    "method": "grid",
    "name": "sst2_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 4},
        "num_train_epochs": {"value": 10},
        "learning_rate": {"max": 8, "min": 4}
    }
}

sweep_configurations["roberta_large_squad_lora_bf16"] = {
    "method": "grid",
    "name": "squad_hyperparameter_sweep",
    "metric": {"goal": "maximize", "name": "f1"},
    "parameters": {
        "per_device_train_batch_size": {"value": 4},
        "num_train_epochs": {"values": 10},
        "learning_rate": {"max": 9, "min": 5}
    }
}

# ========== MINOTAUR Sweep Configurations ==========

sweep_configurations["mobilebert_tiny_mrpc_sgd_lora_bf16"] = {
    "method": "grid",
    "name": "mrpc_lora_sweep_step_lr",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"value": 30},
        "learning_rate": {"max": 15, "min": 6}
    },
}

sweep_configurations["mobilebert_tiny_squad_bf16"] = {
    "method": "grid",
    "name": "squad_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"max": 7, "min": 3},
        "learning_rate": {"max": 5, "min": 1}
    },
}

sweep_configurations["bert_base_squad_posit8"] = {
    "method": "grid",
    "name": "squad_sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "per_device_train_batch_size": {"value": 16},
        "num_train_epochs": {"max": 7, "min": 3},
        "learning_rate": {"max": 9, "min": 5}
    },
}
