#!/usr/bin/env python3
"""
Classify audio files in a folder using PRISM.

Usage:
    python scripts/classify.py \
        --audio_folder path/to/wavs \
        --class_labels class_labels.txt

class_labels.txt should have one class name per line (line 0 = class 0).

Example:
    python scripts/classify.py \
        --audio_folder demo/audio \
        --class_labels demo/us8k_labels.txt
"""

import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laion_clap
from prism import prism, deploy
from prism.prompt_ensemble import get_multi_prompt_text_features


def load_model():
    print("Loading LAION-CLAP...")
    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt()
    model.eval()
    return model


def load_class_labels(path):
    with open(path) as f:
        return {i: line.strip() for i, line in enumerate(f) if line.strip()}


def extract_embeddings(model, wav_files, batch_size=64):
    all_embeds = []
    for i in range(0, len(wav_files), batch_size):
        batch = wav_files[i:i + batch_size]
        with torch.no_grad():
            embds = model.get_audio_embedding_from_filelist(batch)
        all_embeds.append(torch.tensor(embds).float())
    E = torch.cat(all_embeds, dim=0)
    return F.normalize(E, p=2, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_folder', required=True,
                        help='Folder containing .wav files to classify')
    parser.add_argument('--class_labels', required=True,
                        help='Text file with one class name per line')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    label_map = load_class_labels(args.class_labels)
    n_classes = len(label_map)

    wav_files = sorted(glob.glob(os.path.join(args.audio_folder, '*.wav')))
    if not wav_files:
        print(f"No .wav files found in {args.audio_folder}")
        return

    print(f"Found {len(wav_files)} files. Classifying with PRISM...")

    model = load_model()
    T = get_multi_prompt_text_features(label_map, model)
    E = extract_embeddings(model, wav_files, args.batch_size)
    del model

    # Calibrate PRISM on the batch (transductive — no labels needed)
    E_denoised, W = prism(E, T, n_classes)

    # Classify
    logits = E_denoised @ T.t()
    preds = logits.argmax(dim=1).numpy()

    print(f"\n{'File':<40s}  {'Prediction':<25s}  {'Confidence':>10s}")
    print("─" * 80)
    for fname, pred_idx in zip(wav_files, preds):
        confs = torch.softmax(logits[list(wav_files).index(fname)] * 20.0, dim=0)
        conf = confs[pred_idx].item() * 100
        print(f"{os.path.basename(fname):<40s}  {label_map[pred_idx]:<25s}  {conf:8.1f}%")

    print(f"\nDone. W matrix saved — reuse with deploy() for new samples:")
    print("  from prism import deploy")
    print("  E_clean = deploy(E_new, W)")


if __name__ == '__main__':
    main()
