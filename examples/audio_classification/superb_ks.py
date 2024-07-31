import argparse
import logging

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from torchaudio.sox_effects import apply_effects_file

from quantized_training import add_qspec_args, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Process model parameters.")
    parser.add_argument('--model_id', help='Fine-tuned model identifier')
    add_qspec_args(parser)
    return parser.parse_args()


def map_to_array(example):
    import soundfile as sf

    speech_array, sample_rate = sf.read(example["file"])
    example["speech"] = speech_array
    example["sample_rate"] = sample_rate
    return example


def sample_noise(example):
    # Use this function to extract random 1 sec slices of each _silence_ utterance,
    # e.g. inside `torch.utils.data.Dataset.__getitem__()`
    from random import randint

    if example["label"] == "_silence_":
        random_offset = randint(0, len(example["speech"]) - example["sample_rate"] - 1)
        example["speech"] = example["speech"][random_offset : random_offset + example["sample_rate"]]

    return example

# effects = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]
# def map_to_array(example):
#     speech, _ = apply_effects_file(example["file"], effects)
#     example["speech"] = speech.squeeze(0).numpy()
#     return example

# def map_to_array(example):
#     import numpy as np
#     from pydub import AudioSegment
#     audio = AudioSegment.from_file(example["file"])
#     audio = audio.set_channels(1)
#     audio = audio.set_frame_rate(16000)
#     audio = audio - 3.0
#     arr = np.array(audio.get_array_of_samples())
#     bit_depth = audio.sample_width * 8
#     example["speech"] = arr.astype(np.float32) / 2 ** (bit_depth-1)
#     return example


@setup_logging
def main(args):
    dataset = load_dataset("superb", "ks", split="test")
    dataset = dataset.map(map_to_array)
    sampling_rate = dataset.features["audio"].sampling_rate

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = AutoModelForAudioClassification.from_pretrained(args.model_id)
    id2label = model.config.id2label
    print(id2label)

    # Map model label to dataset id
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

    def map_to_pred(batch):
        inputs = feature_extractor(batch["speech"], sampling_rate=sampling_rate, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        output = torch.argmax(logits, dim=-1)
        batch["prediction"] = [label2id[id2label[output[i].item()]] for i in range(len(output))]
        return batch

    result = dataset.map(map_to_pred, batched=True, batch_size=16)

    metric = evaluate.load("accuracy")
    logger.info(metric.compute(predictions=result['prediction'], references=result['label']))


if __name__ == "__main__":
    args = parse_args()
    main(args)
