import argparse
import re
import subprocess

models = [
    'models/mobilebert_tiny_squad',
    'csarron/mobilebert-uncased-squad-v1',
    'csarron/bert-base-uncased-squad-v1',
    'bert-large-uncased-whole-word-masking-finetuned-squad',
    "distilbert-base-uncased-distilled-squad",
]

operations = [
    'gemm',
    'gemm,residual',
    'gemm,residual,norm',
    'gemm,residual,norm,act',
    'gemm,residual,norm,act,scaling',
]

dtypes = ['posit8_1', 'e4m3']

template_command = (
    'python task_runner.py --module examples/question_answering/run_qa_no_trainer.py'
    ' --dataset_name squad --max_seq_length 384 --pad_to_max_length'
    ' --model_name_or_path [model] --per_device_eval_batch_size 16 --bf16'
    ' --quantize_weights --quantize_fwd [ops] --dtype [dtype] --log_file [log_file]'
    ' --gpu [gpu]'
)

def extract_scores(filename):
    scores = []
    with open(filename, 'r') as file:
        for line in file:
            if (match := re.search(r"'f1': (\d+\.\d+)", line)) is not None:
                scores.append(float(match.group(1)))
    scores = [scores[i:i+10] for i in range(0, len(scores), 10)]
    with open('accuracy', 'w') as file:
        for batch in scores:
            file.write('\t'.join(map(str, batch)) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='logs/squad.log')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    for model in models:
        for ops in operations:
            for dtype in dtypes:
                command = template_command.replace('[model]', model)
                command = command.replace('[ops]', ops)
                command = command.replace('[dtype]', dtype)
                command = command.replace('[log_file]', args.log_file)
                command = command.replace('[gpu]', args.gpu)
                print(command)
                subprocess.run(command, shell=True, check=True)
                extract_scores(args.log_file)

if __name__ == "__main__":
    main()