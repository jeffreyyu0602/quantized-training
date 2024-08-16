import argparse
import itertools
import os
import re
import subprocess
import sys

import pandas as pd

models = [
    'models/mobilebert_tiny_squad',
    'csarron/mobilebert-uncased-squad-v1',
    "distilbert-base-uncased-distilled-squad",
    'csarron/bert-base-uncased-squad-v1',
    'bert-large-uncased-whole-word-masking-finetuned-squad',
]

operations = [
    'gemm,residual,layernorm,activation,scaling',
    'gemm,residual,layernorm,activation',
    'gemm,residual,layernorm',
    'gemm,residual',
    'gemm',
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
        '--activation', dtype,
        '--weight', dtype,
        '--quantize_forward', ops,
        '--log_file', log_file,
    ]
    if gpu is not None:
        command += ['--gpu', gpu]
    print("Running:", ' '.join(command))
    subprocess.run(command, check=True)


def extract_f1_scores(log_file, out_file):
    with open(log_file, 'r') as file, open(out_file + '.out', 'w') as out:
        scores = (re.findall(r"'f1': (\d+\.\d+)", file.read()))
        for i in range(0, len(scores), 10):
            out.write('\t'.join(scores[i:i+10]) + '\n')
        return scores


def write_csv(scores, out_file):
    assert len(scores) == 50, "Expected 50 scores, got %d" % len(scores)

    rows = [
        'MobileBERT-tiny',
        'MobileBERT',
        'DistillBERT-base',
        'BERT-base',
        'BERT-large'
    ]
    headers = [
        "no fusion",
        "gemm + attention scaling",
        "\'+ activation fusion",
        "\'+ layernorm fusion",
        "\'+ residual fusion"
    ]
    subheaders = ['Posit8', 'E4M3']

    columns = pd.MultiIndex.from_product([headers, subheaders],
                                         names=['Fusion', 'Data Type'])

    scores_matrix = [scores[i:i+10] for i in range(0, len(scores), 10)]
    df = pd.DataFrame(scores_matrix, index=rows, columns=columns)
    df.to_csv(out_file + '.csv')


def run_experiments(args):
    for ops, model, dtype in itertools.product(operations, models, dtypes):
        run_evaluation(model, ops, dtype, args.log_file, args.gpu)
        scores = extract_f1_scores(args.log_file, args.out_file)
    print("All commands executed.")

    rows = [
        "no fusion",
        "gemm + attention scaling",
        "\'+ activation fusion",
        "\'+ layernorm fusion",
        "\'+ residual fusion"
    ]
    headers = [
        'MobileBERT-tiny',
        'MobileBERT',
        'BERT-base',
        'BERT-large',
        'DistillBERT-base'
    ]
    subheaders = ['Posit8', 'E4M3']

    columns = pd.MultiIndex.from_product([headers, subheaders],
                                         names=['Model', 'Data Type'])

    scores_matrix = [scores[i:i+10] for i in range(0, len(scores), 10)]
    df = pd.DataFrame(scores_matrix, index=rows, columns=columns)
    df.to_csv('squad_f1.csv')


def run_experiments_v2(args):
    for model, ops, dtype in itertools.product(models, operations, dtypes):
        run_evaluation(model, ops, dtype, args.log_file, args.gpu)
        scores = extract_f1_scores(args.log_file, args.out_file)
    print("All commands executed.")
    write_csv(scores, args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='logs/squad.log')
    parser.add_argument('--out_file', default='squad_f1')
    parser.add_argument('--gpu', default=None)
    args = parser.parse_args()

    if os.path.exists(args.log_file) and os.path.getsize(args.log_file) > 0:
        print("Log file exists and is not empty. Extracting scores...")
        scores = extract_f1_scores(args.log_file, args.out_file)
        write_csv(scores, args.out_file)
        sys.exit(0)

    run_experiments_v2(args)
