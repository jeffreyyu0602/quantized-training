import os
import re
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import make_interp_spline

logger = logging.getLogger(__name__)

def get_grouped_histogram(model):
    layer_groups = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, torch.ao.quantization.FakeQuantizeBase):
            if (match := re.search(r'(.*?\.\d+)\.(.*?)?(?=\.activation_pre_process\.\d+)', name)):
                prefix, layer_name = match.group(1), match.group(2)
            else:
                prefix = layer_name = re.sub(r'\.activation_pre_process\.\d+', '', name)
            layer_groups[prefix].append((layer_name, module.histogram.cpu().numpy()))
    return layer_groups

def plot_histogram(model, output_dir):
    histc = get_grouped_histogram(model)
    for layer_name, histograms in histc.items():
        hist_sum = np.sum(np.stack([hist for _, hist in histograms]), axis=0)
        cumulative_sum = np.cumsum(hist_sum) / np.sum(hist_sum)
        min_quantile = np.searchsorted(cumulative_sum, 0.005)

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
        plt.title(f'{layer_name} Histogram')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        plt.subplots_adjust(right=0.75)
        plt.savefig(os.path.join(output_dir, f"{layer_name}.png"))
        plt.close()

def plot_layer_range(model, output_dir):
    histc = get_grouped_histogram(model)
    # Step 1: Aggregate histograms by layer name
    layer_histograms = {}
    for _, histograms in histc.items():
        for name, hist in histograms:
            layer_name = name.replace('activation_pre_process.', '')
            layer_histograms.setdefault(layer_name, np.zeros_like(hist))
            layer_histograms[layer_name] += hist

    # Calculate the overall histogram and determine global quantile values
    hist_sum = np.sum(np.stack(list(layer_histograms.values())), axis=0)
    cumulative_sum = np.cumsum(hist_sum) / np.sum(hist_sum)
    min_quantile = np.searchsorted(cumulative_sum, 0.005) - 126

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
    plt.savefig(os.path.join(output_dir, "layer_distribution.png"), dpi=300)
    plt.close()
