import argparse
import json
import tempfile
from terminaltables import AsciiTable

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.library import Library, impl
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor, BeitForSemanticSegmentation
from huggingface_hub import hf_hub_download

from metrics import eval_metrics
from quantized_training import add_training_args, quantize_fx


class AlignedResize:
    def __init__(self, scale, size_divisor):
        self.scale = scale
        self.size_divisor = size_divisor

    def __call__(self, image):
        w, h = image.size

        max_long_edge = max(self.scale)
        max_short_edge = min(self.scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
        new_w, new_h = int(w * float(scale_factor) +
                           0.5), int(h * float(scale_factor) + 0.5)

        align_h = int(np.ceil(new_h / self.size_divisor)) * self.size_divisor
        align_w = int(np.ceil(new_w / self.size_divisor)) * self.size_divisor

        # image = v2.functional.resize(image, (align_h, align_w))
        image = v2.functional.resize(image, (align_h, align_w))
        return image


def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


def parse_args():
    parser = argparse.ArgumentParser(description="Process model parameters.")
    parser.add_argument(
        '--model_id', required=True, help='Fine-tuned model identifier'
    )
    add_training_args(parser)
    return parser.parse_args()


def main(args):
    dataset = load_dataset("scene_parse_150", split="validation")

    test_pipeline = v2.Compose([
        # AlignedResize(scale=(512.0, 2048.0), size_divisor=32),
        v2.Resize((512, 21 * 32)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float),
        v2.Normalize(mean=[123.675, 116.28, 103.53],
                     std=[58.395, 57.12, 57.375]),
    ])

    # beit_pipeline = v2.Compose([
    #     v2.Resize((640, 640)),
    #     v2.PILToTensor(),
    #     v2.ToDtype(torch.float),
    #     v2.Normalize(mean=[123.675, 116.28, 103.53],
    #                  std=[58.395, 57.12, 57.375]),
    # ])

    def transform(example_batch):
        example_batch["pixel_values"] = [
            test_pipeline(x).unsqueeze(0) for x in example_batch["image"]
        ]
        # example_batch["pixel_values"] = [
        #     beit_pipeline(x).unsqueeze(0) for x in example_batch["image"]
        # ]
        # print(beit_pipeline(example_batch["image"][0]).unsqueeze(0).shape)
        # print(image_processor(example_batch["image"][0]).pixel_values[0].shape)
        # example_batch["pixel_values"] = [
        #     image_processor(x, return_tensors="pt").pixel_values for x in example_batch["image"]
        # ]
        example_batch["label"] = [
            v2.functional.pil_to_tensor(x) for x in example_batch["annotation"]
        ]
        return example_batch

    dataset.set_transform(transform)

    id2label = json.load(open(hf_hub_download(
        "huggingface/label-files", "ade20k-id2label.json", repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    class_names = list(id2label.values())
    num_classes = len(class_names)

    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")

    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_id).to(device)
    image_processor = AutoImageProcessor.from_pretrained(args.model_id)

    m = Library("my_custom_library", "DEF")

    m.define("my_interpolate(Tensor input, SymInt[] size, float[]? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None, bool antialias = False) -> Tensor")

    orig_interpolate = F.interpolate

    @impl(m, "my_interpolate", "CompositeExplicitAutograd")
    def my_interpolate(*args, **kwargs):
        return orig_interpolate(*args, **kwargs)

    F.interpolate = torch.ops.my_custom_library.my_interpolate

    example_args = (dataset[0]["pixel_values"].to(device),)
    model = quantize_fx(model, args, example_args)

    # model.graph.print_tabular()

    results = []
    gt_seg_maps = []
    for data in tqdm(dataset):
        pixel_values = data["pixel_values"].to(device)
        with torch.no_grad():
            result = model(pixel_values)

        logit = F.interpolate(
            result["logits"], size=data["label"].shape[-2:], mode="bilinear"
        )

        results.append(logit.argmax(dim=1).cpu())
        gt_seg_maps.append(data["label"])

    ret_metrics = eval_metrics(
        results,
        gt_seg_maps,
        num_classes,
        ignore_index=255,
        reduce_zero_label=True
    )

    class_table_data = [['Class', 'IoU', 'Acc']]
    ret_metrics_round = [
        np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
    ]
    for i in range(num_classes):
        class_table_data.append([class_names[i]] +
                                [m[i] for m in ret_metrics_round[2:]] +
                                [ret_metrics_round[1][i]])
    summary_table_data = [['Scope'] +
                          ['m' + head for head in class_table_data[0][1:]] +
                          ['aAcc']]
    ret_metrics_mean = [
        np.round(np.nanmean(ret_metric) * 100, 2) for ret_metric in ret_metrics
    ]
    summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                              [ret_metrics_mean[1]] +
                              [ret_metrics_mean[0]])
    print('per class results:')
    table = AsciiTable(class_table_data)
    print('\n' + table.table)
    print('Summary:')
    table = AsciiTable(summary_table_data)
    print('\n' + table.table)


if __name__ == "__main__":
    args = parse_args()
    main(args)
