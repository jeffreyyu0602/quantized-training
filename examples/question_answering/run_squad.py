import argparse
import os
import re
import subprocess
import sys
import pandas as pd

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
        return scores

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--log_file', default='logs/squad.log')
#     parser.add_argument('--out_file', default='accuracy.out')
#     parser.add_argument('--gpu', default=None)
#     args = parser.parse_args()

#     for model in models:
#         for ops in operations:
#             for dtype in dtypes:
#                 run_evaluation(model, ops, dtype, args.log_file, args.gpu)
#                 extract_scores(args.log_file, args.out_file)

#     print("All commands executed.")

def run_asplos_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='logs/squad.log')
    parser.add_argument('--out_file', default='accuracy.out')
    parser.add_argument('--gpu', default=None)
    args = parser.parse_args()

    if os.path.exists(args.log_file) and os.path.getsize(args.log_file) > 0:
        print("Log file exists and is not empty. Exiting program.")
        sys.exit(1)

    operations.reverse()
    for ops in operations:
        for model in models:
            for dtype in dtypes:
                run_evaluation(model, ops, dtype, args.log_file, args.gpu)
                scores = extract_scores(args.log_file, args.out_file)

    print("All commands executed.")

    rows = [
        "no fusion", 
        "fuse gemm + attention scaling", 
        "+ activation fusion", 
        "+ layernorm fusion", 
        "+ residual fusion"
    ]

    columns = pd.MultiIndex.from_product([
        ['MobileBERT-tiny', 'MobileBERT', 'BERT-base', 'BERT-large', 'DistillBERT-base'],  # Main headers
        ['Posit8', 'FP8']  # Sub-headers
    ], names=['Model', 'Precision'])

    scores_matrix = [scores[i:i+10] for i in range(0, len(scores), 10)]
    df = pd.DataFrame(scores_matrix, index=rows, columns=columns)
    df.to_csv('squad_f1.csv')

if __name__ == "__main__":
    run_asplos_experiments()