import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from evaluate import load

def read_metrics(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    metrics = []
    num_words = []
    for line in lines:
        if (match := re.match(r'^(\d+\.\d+)\s+(\d+)$', line)) is not None:
            metrics.append(float(match.group(1)))
            num_words.append(int(match.group(2)))

    return metrics, num_words

def compute_wer(ref_file, pred_file):
    with open(ref_file, 'r') as f:
        references = f.readlines()

    with open(pred_file, 'r') as f:
        predictions = f.readlines()

    assert len(references) == len(predictions), "Number of lines do not match"

    wer = load("wer")
    wer_scores = []
    num_words = []
    for i in range(len(references)):
        wer_scores.append(100 * wer.compute(references=[references[i]], predictions=[predictions[i]]))
        num_words.append(len(references[i].split()))

    print(sum([a * b for a, b in zip(wer_scores, num_words)]) / sum(num_words))

    print(100 * wer.compute(references=references, predictions=predictions))

    return wer_scores, num_words, references, predictions

def generate_hist(ref_file, pred_file1, pred_file2, output_dir):
    wer_scores1, num_words1, ref1, pred1 = compute_wer(ref_file, pred_file1)
    wer_scores2, num_words2, ref2, pred2 = compute_wer(ref_file, pred_file2)
    total_words = sum(num_words1)

    diffs = []
    for i in range(len(wer_scores1)):
        diffs.append((wer_scores1[i] - wer_scores2[i]) * num_words1[i] / total_words)

    with open(os.path.join(output_dir, "wer_diff.txt"), "w") as f:
        for i in range(len(diffs)):
            indicator = "" if abs(diffs[i]) < 0.1 else " ***"
            f.write(f"{int(num_words1[i]):3d}\t{wer_scores1[i]:6.2f}\t{wer_scores2[i]:6.2f}\t{diffs[i]:9.6f}{indicator}\n")

            if abs(diffs[i]) > 0.1:
                f.write(ref1[i])
                f.write(pred1[i])
                f.write(pred2[i])

    n, bins, patches = plt.hist(diffs, bins=np.arange(-1, 1, 0.05))

    for i in range(len(n)):
        if n[i] > 0:
            plt.text(bins[i], n[i], str(int(n[i])), ha='center', va='bottom')

    plt.xlabel('WER score')
    plt.ylabel('Number of samples')
    plt.title('WER Differences')
    plt.savefig(os.path.join(output_dir, "wer_hist.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process metrics files.")
    parser.add_argument('ref_file', type=str, help='Path to the first metrics file')
    parser.add_argument('pred_file1', type=str, help='Path to the second metrics file (optional)')
    parser.add_argument('pred_file2', type=str, help='Path to the second metrics file (optional)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for histogram.')
    args = parser.parse_args()
    generate_hist(args.ref_file, args.pred_file1, args.pred_file2, args.output_dir)