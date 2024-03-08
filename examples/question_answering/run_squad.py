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

ablation_study = [
    'gemm',
    'gemm,residual',
    'gemm,norm',
    'gemm,act',
    'gemm,scaling',
]

dtypes = ['posit8_1', 'e4m3']

def run_evaluation(model, ops, dtype, log_file, gpu):
    command = [
        'python', 'examples/question_answering/run_qa_no_trainer.py',
        '--model_name_or_path', model, 
        '--dataset_name', 'squad', 
        '--per_device_eval_batch_size', '16', 
        '--max_seq_length', '384', 
        '--doc_stride', '128',
        '--pad_to_max_length',
        '--bf16',
        '--quantize_weights', 
        '--quantize_fwd', ops, 
        '--dtype', dtype, 
        '--log_file', log_file,
    ]
    if gpu is not None:
        command += ['--gpus', gpu]
    print("Running:", ' '.join(command))
    subprocess.run(command, check=True)

def extract_scores(log_file, out_file):
    with open(log_file, 'r') as file, open(out_file, 'w') as out:
        scores = (re.findall(r"'f1': (\d+\.\d+)", file.read()))
        for i in range(0, len(scores), 10):
            out.write('\t'.join(scores[i:i+10]) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='logs/squad.log')
    parser.add_argument('--out_file', default='accuracy.out')
    parser.add_argument('--gpu', default=None)
    args = parser.parse_args()

    for model in models:
        for ops in operations:
            for dtype in dtypes:
                run_evaluation(model, ops, dtype, args.log_file, args.gpu)
                extract_scores(args.log_file, args.out_file)

    print("All commands executed.")

if __name__ == "__main__":
    main()