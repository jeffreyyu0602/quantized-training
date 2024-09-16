import argparse
import copy
import hashlib
import logging
import os
import random
import re
from random import randint

import evaluate
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
from datasets import DatasetDict, Dataset, load_dataset
from scipy.fftpack import dct
from tqdm import tqdm
from transformers import set_seed

from honk_model import SpeechResModel, configs
from quantized_training import (
    add_qspec_args,
    get_default_quantizer,
    prepare_pt2e,
    quantize,
    setup_logging,
)

logger = logging.getLogger(__name__)

bg_noise_audio = []
unknown_prob = 0.1
silence_prob = 0.1
noise_prob = 0.8
input_length = 16000
timeshift_ms = 100
_audio_cache = {}
_file_cache = {}

label2id = {
    'yes': 0,
    'no': 1,
    'up': 2,
    'down': 3,
    'left': 4,
    'right': 5,
    'on': 6,
    'off': 7,
    'stop': 8,
    'go': 9,
    '_silence_': 10,
    '_unknown_': 11,
}


MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'test'
  else:
    result = 'train'
  return result


def load_local_dataset(data_dir):
    raw_datasets = {
        "train": {"file": [], "label": [], "unknown": []},
        "validation": {"file": [], "label": [], "unknown": []},
        "test": {"file": [], "label": [], "unknown": []},
    }
    bg_noise_files = []
    unknown_label = label2id["_unknown_"]
    for folder_name in os.listdir(data_dir):
        path_name = os.path.join(data_dir, folder_name)
        if os.path.isfile(path_name):
            continue
        label = label2id.get(folder_name, unknown_label)

        for filename in os.listdir(path_name):
            if not filename.endswith(".wav"):
                continue
            wav_name = os.path.join(path_name, filename)
            if folder_name == "_background_noise_":
                bg_noise_files.append(wav_name)
            else:
                set_name = which_set(str(wav_name), 10, 10)
                if label == unknown_label:
                    raw_datasets[set_name]["unknown"].append(wav_name)
                else:
                    raw_datasets[set_name]["file"].append(wav_name)
                    raw_datasets[set_name]["label"].append(label)

    # unknown_files = (raw_datasets["train"]["unknown"] +
    #                  raw_datasets["validation"]["unknown"] +
    #                  raw_datasets["test"]["unknown"])
    for dataset_dict in raw_datasets.values():
        num_examples = len(dataset_dict["file"])

        # num_unknown_samples = int(unknown_prob * num_examples)
        # dataset_dict["file"] += random.sample(dataset_dict["unknown"], num_unknown_samples)
        # dataset_dict["label"] += [unknown_label] * num_unknown_samples

        # choose from the entire unknown set
        # dataset_dict["file"] += random.sample(unknown_files, num_unknown_samples)
        # unknown_files = [f for f in unknown_files if f not in dataset_dict["file"]]
        # dataset_dict["label"] += [unknown_label] * num_unknown_samples

        # add all unknown samples
        dataset_dict["file"] += dataset_dict["unknown"]
        dataset_dict["label"] += [unknown_label] * len(dataset_dict["unknown"])
        del dataset_dict["unknown"]

        num_silence_samples = int(silence_prob * num_examples)
        dataset_dict["file"] += ["silence"] * num_silence_samples
        dataset_dict["label"] += [label2id["_silence_"]] * num_silence_samples

    raw_datasets = DatasetDict({
        "train": Dataset.from_dict(raw_datasets["train"]),
        "validation": Dataset.from_dict(raw_datasets["validation"]),
        "test": Dataset.from_dict(raw_datasets["test"]),
    })

    # FIXME: avoid using global variables
    global bg_noise_audio
    bg_noise_audio = [librosa.core.load(file, sr=16000)[0] for file in bg_noise_files]

    return raw_datasets, bg_noise_files


def _timeshift_audio(data):
    shift = (16000 * timeshift_ms) // 1000
    shift = randint(-shift, shift)
    a = -min(0, shift)
    b = max(0, shift)
    data = np.pad(data, (a, b), "constant")
    return data[:len(data) - a] if a else data[b:]


def _compute_mfcc(data):
    data = librosa.feature.melspectrogram(
        y=data,
        sr=16000,
        n_mels=40,
        hop_length=160,
        n_fft=480,
        fmin=20,
        fmax=4000,
        pad_mode='reflect',
    )
    data[data > 0] = np.log(data[data > 0])
    dct_filters = dct(np.eye(40), type=2, norm='ortho').T
    data = [np.matmul(dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
    data = np.array(data, order="F").astype(np.float32)
    return data


def _log_melspectrogram(array):
    S = librosa.feature.melspectrogram(
        y=array, n_fft=400, hop_length=160, sr=16000, n_mels=40
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


def prepare_audio(example, train=False):
    if example["file"] == "silence":
        data = np.zeros(input_length, dtype=np.float32)
    elif (data := _file_cache.get(example["file"])) is None:
        data = librosa.core.load(example["file"], sr=16000)[0]
        _file_cache[example["file"]] = data

    if len(data) > input_length:
        i = randint(0, len(data) - input_length - 1)
        data = data[i:i + input_length]
    elif len(data) < input_length:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    if train:
        # performs a random time-shift of 100 milliseconds before transforming the audio into MFCCs
        data = _timeshift_audio(data)

    if len(bg_noise_audio) > 0:
        # Adds background noise to each sample with a probability of 0.8 at every epoch
        # TODO: the original code add different randome noise to silence samples for evaluation as well
        if (train and random.random() < noise_prob) or example["file"] == "silence":
            bg_noise = random.choice(bg_noise_audio)
            i = random.randint(0, len(bg_noise) - input_length - 1)
            bg_noise = bg_noise[i:i + input_length]
            data = np.clip(random.random() * 0.1 * bg_noise + data, -1, 1)
    # example["audio"] = _compute_mfcc(data).reshape(-1, 40)
    example["audio"] = _log_melspectrogram(data)
    return example


def collate_fn(data):
    input_values = torch.from_numpy(
        np.stack([prepare_audio(x, train=True)["audio"] for x in data])
    )
    labels = torch.tensor([x["label"] for x in data])
    return {'input_values': input_values, 'labels': labels}


def parse_args():
    parser = argparse.ArgumentParser(description="Process model parameters.")
    parser.add_argument('--model_id', default=None, help='Path to fine-tuned model.')
    parser.add_argument('--output_file', default="res8-narrow.pt", help='Path to save the best model.')
    parser.add_argument('--data_dir', default=None, help='Path to the Google speech command dataset.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_train_epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--qat', action='store_true', help='Perform quantization-aware training.')
    add_qspec_args(parser)
    return parser.parse_args()


@setup_logging
def main(args):
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic=True

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
    else:
        logger.warning("CUDA is not available.")
        device = torch.device("cpu")

    if args.data_dir is not None:
        raw_dataset, _ = load_local_dataset(args.data_dir)
    else:
        raw_dataset = load_dataset("superb", "ks")

    train_dataloader = torch.utils.data.DataLoader(
        raw_dataset["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    raw_dataset["validation"] = raw_dataset["validation"].map(prepare_audio)
    raw_dataset["test"] = raw_dataset["test"].map(prepare_audio)

    model = SpeechResModel(configs["res8-narrow"])
    model.to(device)

    if args.qat:
        modules_to_fuse = [[f'conv{i}', f'bn{i}'] for i in range(1, 7)]
        model = torch.ao.quantization.fuse_modules_qat(model, modules_to_fuse)

        quantize(model, args)
        batch = next(iter(train_dataloader))
        with torch.no_grad():
            model(batch["input_values"].to(device))

    if args.model_id is not None:
        checkpoint = torch.load(args.model_id, map_location=device)
        model.load_state_dict(checkpoint)

    if args.pt2e_quantization:
        print("PyTorch 2 Export quantization")
        modules_to_fuse = [[f'conv{i}', f'bn{i}'] for i in range(1, 7)]
        model = torch.ao.quantization.fuse_modules(model.eval(), modules_to_fuse)

        quantizer = get_default_quantizer(
            input_activation=args.activation,
            weight=args.weight,
            record_histogram=args.record_histogram,
            force_scale_power_of_two=args.force_scale_power_of_two,
        )
        batch = next(iter(train_dataloader))
        example_args = ({k: v.to(device) for k, v in batch.items()},)
        dynamic_shapes = {"x": {0: torch.export.Dim("batch")}}
        model = prepare_pt2e(model, quantizer, example_args, dynamic_shapes=dynamic_shapes)

        def calibrate(model):
            for batch in tqdm(train_dataloader):
                with torch.no_grad():
                    model(batch["input_values"].to(device))
        calibrate(model)

        for module in model.modules():
            if isinstance(module, torch.ao.quantization.FakeQuantizeBase):
                module.disable_observer()

    def map_to_pred(batch):
        input_values = torch.tensor(batch["audio"])
        with torch.no_grad():
            logits = model(input_values.to(device))
        batch["prediction"] = torch.argmax(logits, dim=-1)
        return batch

    metric = evaluate.load("accuracy")

    if not args.do_train:
        model.eval()
        result = raw_dataset["test"].map(
            map_to_pred, batched=True, batch_size=args.batch_size, desc="Test"
        )
        eval_metric = metric.compute(predictions=result['prediction'], references=result['label'])
        logger.info(f"Test accuracy: {eval_metric['accuracy']}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_epochs, eta_min=0.0001)

    best_acc = 0.0
    best_state = None
    for epoch in range(args.num_train_epochs):
        total_loss = 0.0
        model.train()
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch["input_values"])
            loss = nn.CrossEntropyLoss()(output, batch["labels"])
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        lr_scheduler.step()

        total_loss /= len(train_dataloader)
        logger.info(f"Epoch {epoch}: Loss {total_loss}")

        model.eval()
        result = raw_dataset["validation"].map(
            map_to_pred, batched=True, batch_size=args.batch_size, desc="Validate"
        )
        eval_metric = metric.compute(predictions=result['prediction'], references=result['label'])
        logger.info(f"Epoch {epoch}: Validation accuracy {eval_metric['accuracy']}")

        if eval_metric['accuracy'] > best_acc:
            best_acc = eval_metric['accuracy']
            best_state = copy.deepcopy(model.state_dict())
            if args.output_file is not None:
                torch.save(model.state_dict(), args.output_file)

    logger.info(f"Best validation accuracy: {best_acc}")

    model.load_state_dict(best_state)
    model.eval()
    result = raw_dataset["test"].map(
        map_to_pred, batched=True, batch_size=args.batch_size, desc="Test"
    )
    eval_metric = metric.compute(predictions=result['prediction'], references=result['label'])
    logger.info(f"Final test accuracy: {eval_metric['accuracy']}")


if __name__ == "__main__":
    args = parse_args()
    main(args)