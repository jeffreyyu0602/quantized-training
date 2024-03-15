import os
import re
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

logger = logging.getLogger(__name__)

def get_histogram_pre_process(model, prefix):
    histc = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.ao.quantization.FakeQuantizeBase):
            match = re.match(prefix, name)
            key = match.group(1) if match else name.replace('activation_pre_process.', '')
            histc.setdefault(key, [])

            layer_name = re.sub(prefix, '', name)
            layer_name = layer_name.replace('activation_pre_process.', '')
            histc[key].append((layer_name, module.histogram_pre_process.cpu().numpy()))
    return histc

def plot_layer_distribution(model, prefix, output_dir):
    histc = get_histogram_pre_process(model, prefix)
    for index, histograms in histc.items():
        hist_sum = np.sum(np.stack([hist for _, hist in histograms]), axis=0)
        cumulative_sum = np.cumsum(hist_sum) / np.sum(hist_sum)
        min_quantile = np.searchsorted(cumulative_sum, 0.001)
        max_quantile = min(np.searchsorted(cumulative_sum, 0.999), cumulative_sum.shape[0] - 1)

        non_zero_bins = np.nonzero(hist_sum)
        min_bin = max(non_zero_bins[0].min(), min_quantile)
        max_bin = non_zero_bins[0].max()

        plt.figure(figsize=(10, 6))
        for name, hist in histograms:
            bins = np.linspace(min_bin, max_bin, max_bin - min_bin + 1) - 126
            adjusted_histogram = (hist / np.sum(hist))[min_bin:max_bin + 1]

            # Smoothing
            X_Y_Spline = make_interp_spline(bins, adjusted_histogram)
            X_ = np.linspace(bins.min(), bins.max(), 500)
            Y_ = X_Y_Spline(X_)

            plt.plot(X_, Y_, label=name)

        plt.xlabel('Exponent Value')
        plt.ylabel('Frequency')
        plt.title(f'Encoder {index} Value Distribution')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"encoder-{index}.png"))
        plt.close()

def plot_layer_range(model, prefix, output_dir):
    histc = get_histogram_pre_process(model, prefix)
    # Step 1: Aggregate histograms by layer name
    layer_histograms = {}
    for _, histograms in histc.items():
        for name, hist in histograms:
            clean_name = name.replace('activation_pre_process.', '')
            layer_histograms.setdefault(clean_name, np.zeros_like(hist))
            layer_histograms[clean_name] += hist

    # Calculate the overall histogram and determine global quantile values
    hist_sum = np.sum(np.stack(layer_histograms.values()), axis=0)
    cumulative_sum = np.cumsum(hist_sum) / np.sum(hist_sum)
    min_quantile = np.searchsorted(cumulative_sum, 0.001) - 126
    max_quantile = min(np.searchsorted(cumulative_sum, 0.999), cumulative_sum.shape[0] - 1) - 126

    # Step 2: Determine the minimum and maximum bin values for each layer
    layer_ranges = {}
    for name, hist in layer_histograms.items():
        non_zero_bins = np.flatnonzero(hist) - 126
        min_bin = max(min_quantile, non_zero_bins.min())
        max_bin = non_zero_bins.max()
        layer_ranges[name] = (min_bin, max_bin)

    # Step 3: Plot a horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = list(range(len(layer_ranges)))
    heights = [range[1] - range[0] for range in layer_ranges.values()]
    lefts = [range[0] for range in layer_ranges.values()]
    ax.barh(y_pos, heights, left=lefts, height=1, color='tab:blue', edgecolor='black', linewidth=0.5)

    # Set the y-ticks to match the layer names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_ranges.keys())
    ax.set_ylim(min(y_pos)-0.5, max(y_pos)+0.5)

    # Remove all spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Show y-axis at x=0 position without ticks
    ax.axvline(x=0, color='black', linewidth=0.75)
    ax.tick_params(axis='y', which='both', left=False)

    # Labels and title
    ax.set_xlabel('Exponent Value', fontsize=16)

    # Adjust layout to not cut off the longer layer names and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_range.png"), dpi=300)
    plt.close()

def plot_histogram(model, output_dir):
    for name, module in model.named_modules():
        if isinstance(module, torch.ao.quantization.FakeQuantizeBase):
            hist_pre = module.histogram_pre_process.cpu()
            hist_post = module.histogram_post_process.cpu()

            non_empty_bins1 = torch.nonzero(hist_pre).flatten()
            non_empty_bins2 = torch.nonzero(hist_post).flatten()

            if len(non_empty_bins1) == 0 or len(non_empty_bins2) == 0:
                logger.warn("One or both histograms are empty. Skipping plotting.")
                continue

            first_non_zero = min(non_empty_bins1[0], non_empty_bins2[0])
            last_non_zero = max(non_empty_bins1[-1], non_empty_bins2[-1])

            hist_pre = hist_pre[first_non_zero:last_non_zero + 1]
            hist_post = hist_post[first_non_zero:last_non_zero + 1]

            bins = torch.linspace(-126, 127, 255)[first_non_zero:last_non_zero + 2]
            bar_width = (bins[1] - bins[0]) * 0.4

            plt.figure(figsize=(10, 6))
            plt.bar(bins[:-1] - bar_width/2, hist_pre, width=bar_width, label='Before quantization')
            plt.bar(bins[:-1] + bar_width/2, hist_post, width=bar_width, label='After quantization')

            plt.title('Activation Distribution')
            plt.xlabel('Exponent Value')
            plt.ylabel('Count')
            plt.legend()

            plt.savefig(os.path.join(output_dir, f'{name}.png'))
            plt.close()