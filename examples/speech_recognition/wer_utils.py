import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from evaluate import load


def compute_wer(references, predictions):
    assert len(references) == len(predictions), "Number of lines do not match"

    wer = load("wer")
    wer_scores = [100 * wer.compute(references=[ref], predictions=[pred]) for ref, pred in zip(references, predictions)]
    num_words = [len(ref.split()) for ref in references]

    # compute WER by averaging over all score
    print(sum([a * b for a, b in zip(wer_scores, num_words)]) / sum(num_words))

    # compute WER by directly passing predictions and references to the WER metric
    wer = load("wer")
    print(100 * wer.compute(references=references, predictions=predictions))

    return wer_scores, num_words

def compute_differences(references, predictions1, predictions2, output_dir):
    wer_scores1, num_words = compute_wer(references, predictions1)
    wer_scores2, _ = compute_wer(references, predictions2)

    total_words = sum(num_words)
    diffs = [(wer_scores1[i] - wer_scores2[i]) * num_words[i] / total_words for i in range(len(wer_scores1))]

    with open(os.path.join(output_dir, "wer_scores.txt"), "w") as f:
        for i, diff in enumerate(diffs):
            indicator = "" if abs(diff) < 0.1 else " ***"
            f.write(f"{wer_scores1[i]:6.2f}\t{wer_scores2[i]:6.2f}{indicator}\n")
            if abs(diff) > 0.1:
                f.write(f"{references[i]}\n{predictions1[i]}\n{predictions2[i]}\n")

    return diffs

def plot_differences(diffs, output_dir):
    plt.figure()
    n, bins, patches = plt.hist(diffs, bins=np.arange(-1, 1, 0.05))
    for i, count in enumerate(n):
        if count > 0:
            plt.text(bins[i], count, str(int(count)), ha='center', va='bottom')
    plt.xlabel('WER score')
    plt.ylabel('Number of samples')
    plt.title('WER Differences')
    plt.savefig(os.path.join(output_dir, "wer_scores.png"))

def main():
    parser = argparse.ArgumentParser(description="Process metrics files.")
    parser.add_argument('ref_file', help='Path to the reference file')
    parser.add_argument('pred_file1', help='Path to the first predictions file')
    parser.add_argument('pred_file2', help='Path to the second predictions file')
    parser.add_argument('--output_dir', default="./", help='Output directory for results')
    args = parser.parse_args()

    with open(args.ref_file, 'r') as f:
        references = f.readlines()
    with open(args.pred_file1, 'r') as f:
        predictions1 = f.readlines()
    with open(args.pred_file2, 'r') as f:
        predictions2 = f.readlines()

    diffs = compute_differences(references, predictions1, predictions2, args.output_dir)
    plot_differences(diffs, args.output_dir)

if __name__ == "__main__":
    main()