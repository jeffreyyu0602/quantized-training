import argparse
import logging
import os
import random
from random import randint

import evaluate
import librosa
import numpy as np
import torch
from datasets import load_dataset
from scipy.fftpack import dct
from torch import nn
from tqdm import tqdm

from honk_model import SpeechResModel, configs
from quantized_training import add_training_args, run_task

logger = logging.getLogger(__name__)

timeshift_ms = 100
in_len = 16000
noise_prob = 0.8
_file_cache = {}
_audio_cache = {}

noise_dir = "_background_noise_"
bg_noise_files = (
    [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]
    if os.path.exists(noise_dir) else []
)
bg_noise_audio = [librosa.core.load(file, sr=16000)[0] for file in bg_noise_files]


def parse_args():
    parser = argparse.ArgumentParser(description="Process model parameters.")
    parser.add_argument('--model_id', default=None, help='Fine-tuned model identifier')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_train_epochs', type=int, default=30)
    add_training_args(parser)
    return parser.parse_args()


def _timeshift_audio(data):
    shift = (16000 * timeshift_ms) // 1000
    shift = randint(-shift, shift)
    a = -min(0, shift)
    b = max(0, shift)
    data = np.pad(data, (a, b), "constant")
    return data[:len(data) - a] if a else data[b:]


def compute_mfccs(example, train=False):
    if (data := _file_cache.get(example["file"])) is None:
        data = librosa.core.load(example["file"], sr=16000)[0]
        _file_cache[example["file"]] = data
    # truncate noise to 1 second
    if len(data) > in_len:
        a = randint(0, len(data) - in_len - 1)
        data = data[a:a + in_len]
    data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
    if train:
        # performs a random time-shift of 100 milliseconds before transforming the audio into MFCCs
        data = _timeshift_audio(data)

        # Adds background noise to each sample with a probability of 0.8 at every epoch
        if random.random() < noise_prob or example["label"] == 10:
            bg_noise = random.choice(bg_noise_audio)
            i = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[i:i + in_len]
            a = random.random() * 0.1
            data = np.clip(a * bg_noise + data, -1, 1)

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
    example["speech"] = data.reshape(-1, 40)
    return example


def collate_fn(data):
    input_values = torch.from_numpy(
        np.stack([compute_mfccs(x, train=True)["speech"] for x in data])
    )
    labels = torch.tensor([x["label"] for x in data])
    return {'input_values': input_values, 'labels': labels}


def main(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
    else:
        logger.warn("CUDA is not available.")
        device = torch.device("cpu")

    raw_dataset = load_dataset("superb", "ks")
    raw_dataset["validation"] = raw_dataset["validation"].map(compute_mfccs)
    raw_dataset["test"] = raw_dataset["test"].map(compute_mfccs)

    train_dataloader = torch.utils.data.DataLoader(
        raw_dataset["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    model = SpeechResModel(configs["res8-narrow"])
    model.to(device)

    def map_to_pred(batch):
        input_values = torch.tensor(batch["speech"])
        with torch.no_grad():
            logits = model(input_values.to(device))
        batch["prediction"] = torch.argmax(logits, dim=-1)
        # FIXME: model trained using the honk repo has a different label mapping
        # logits = torch.argmax(logits, dim=-1) - 2
        # batch["prediction"] = torch.where(logits < 0, logits + 12, logits)
        return batch

    metric = evaluate.load("accuracy")

    if not args.do_train:
        model.load(args.model_id)
        model.eval()
        result = raw_dataset["test"].map(map_to_pred, batched=True, batch_size=16)
        eval_metric = metric.compute(predictions=result['prediction'], references=result['label'])
        logger.info(f"Test accuracy {eval_metric['accuracy']}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    best_acc = 0.0
    for epoch in range(args.num_train_epochs):
        total_loss = 0.0
        model.train()
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch["input_values"])
            loss = nn.CrossEntropyLoss()(output, batch["labels"])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch == 10:
            lr_scheduler.step()

        total_loss /= len(train_dataloader)
        logger.info(f"Epoch {epoch}: Loss {total_loss}")

        model.eval()
        result = raw_dataset["test"].map(map_to_pred, batched=True, batch_size=16)
        eval_metric = metric.compute(predictions=result['prediction'], references=result['label'])
        logger.info(f"Epoch {epoch}: Validation accuracy {eval_metric['accuracy']}")

        if eval_metric['accuracy'] > best_acc:
            best_acc = eval_metric['accuracy']
            model.save("res8-narrow.pt")


if __name__ == "__main__":
    args = parse_args()
    run_task(main, args)
