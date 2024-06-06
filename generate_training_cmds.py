from itertools import product


models = [
    'models/mobilebert_tiny',
    'google/mobilebert-uncased',
    'roberta-base',
    'roberta-large',
]

tasks = ["mnli", "qnli", "mrpc", "sst2", "squad"]

configs = ["bf16", "posit8", "posit8-approx-shifted", "fp8"]


model_to_filename = {
    "models/mobilebert_tiny": "mobilebert_tiny",
    "google/mobilebert-uncased": "mobilebert_uncased",
    "roberta-base": "roberta_base",
    "roberta-large": "roberta_large",
    "roberta-large-mnli": "roberta_large",
}


def generate_commands():
    with open('asplos_training.sh', 'w') as file:
        for model, task, config in product(models, tasks, configs):
            model_id = (
                "roberta-large-mnli"
                if model == "roberta-large" and task == "mrpc"
                else model
            )
            cmd = [
                "python", "run_asplos_training.py",
                "--model", model_id,
                "--task", task,
                "--run_job", config,
                "--log_file", f"logs/{model_to_filename[model]}-{task}-{config}.log",
            ]
            if model in ["google/mobilebert-uncased", "models/mobilebert_tiny"] and task == "squad":
                cmd += ["--sgd", "--op_fusion", "qa_outputs"]
            if model == "google/mobilebert-uncased" and task != "squad":
                cmd += ["--op_fusion", "classifier"]

            base_cmd = " ".join(cmd)
            for seed in range(3):
                file.write(f"{base_cmd} --seed {seed}\n")
        file.write("\n")


if __name__ == "__main__":
    generate_commands()
