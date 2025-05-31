import argparse
import os
import torch

from torchvision import datasets, transforms
from tqdm import tqdm

from datasets import load_dataset
from .utils import write_tensor_to_file

def get_transforms(model_type):
    if model_type == "resnet":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x
                ),  # Convert grayscale to 3-channel
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif model_type == "vit":
        return transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x
                ),  # Convert grayscale to 3-channel
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def retrieve_daataset(num_samples, model_type):
    transform = get_transforms(model_type)

    # Load the ImageNet dataset from Hugging Face
    dataset = load_dataset("timm/imagenet-1k-wds", split="validation", streaming=True)
    dataset = dataset.take(num_samples)
    
    processed_dataset = []

    # Iterate over the selected indices and retrieve samples
    for i, item in tqdm(enumerate(dataset), desc="Retrieving dataset"):
        image = transform(item["jpg"]).unsqueeze(0)
        label = item["cls"]
        processed_dataset.append({"image": image, "label": label})
        
    return processed_dataset

def dump_imagenet(output_dir, dataset, model_type, preprocess_fn, torch_dtype):
    preprocessed_dataset = []
    for i, image_label_pair in enumerate(tqdm(dataset, desc="Dumping dataset")):
        label = image_label_pair["label"]
        image = image_label_pair["image"]
        dir_name = os.path.join(output_dir, f"{i}_{label}")
        os.makedirs(dir_name, exist_ok=True)
        image = preprocess_fn(image)

        filename = (
            "x_preprocess.bin"
            if model_type == "resnet"
            else "pixel_values_preprocess.bin"
        )

        preprocessed_dataset.append({"image": image.to(torch_dtype), "label": label})
        write_tensor_to_file(image, os.path.join(dir_name, filename))
    return preprocessed_dataset


def main():
    torch.set_num_threads(32)

    parser = argparse.ArgumentParser(
        description="ResNet data generator (from model and images to binary data)."
    )

    parser.add_argument(
        "--output_dir",
        default="data/imagenet",
        help="Path to output folder for dataset.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="How many samples to generate.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["resnet", "vit"],
        help="Model type to use for preprocessing.",
    )
    args = parser.parse_args()
    dump_imagenet(args.output_dir, args.num_samples, args.model_type)


if __name__ == "__main__":
    main()
