#!/usr/bin/env python3
"""
Extract LAION-CLAP audio embeddings from a directory of .wav files.

Produces a .pt file containing a dict:
    { filename: {"class_gt": int, "fold": int, "embd": tensor} }

This format is the input expected by examples/example_us8k.py and
examples/example_esc50.py.

Usage:
    python scripts/extract_embeddings.py \
        --audio_dir ./data/input/urbansound8k_noisy_street_traffic \
        --output_path ./data/embeddings/us8k_noisy_street_traffic.pt \
        --dataset us8k

    python scripts/extract_embeddings.py \
        --audio_dir ./data/input/esc50_noisy_airport \
        --output_path ./data/embeddings/esc50_noisy_airport.pt \
        --dataset esc50

Requirements:
    pip install laion-clap torch tqdm
    # Model checkpoint (630k-audioset-fusion-best.pt) is downloaded automatically
    # by model.load_ckpt() on first run (~600 MB).
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

import laion_clap


def load_model():
    """Load LAION-CLAP with fusion (checkpoint: 630k-audioset-fusion-best.pt)."""
    print("Loading LAION-CLAP model (downloads checkpoint on first run)...")
    model = laion_clap.CLAP_Module(enable_fusion=True, amodel='HTSAT-base')
    model.load_ckpt()  # downloads 630k-audioset-fusion-best.pt automatically
    model.eval()
    return model


def parse_us8k_filename(file_path):
    """
    Parse UrbanSound8K filename convention:
        [fsID]-[classID]-[occurrenceID]-[sliceID].wav
    Returns (class_id, fold) — fold is from the parent directory name.
    """
    file_name = os.path.basename(file_path)
    parts = file_name.replace(".wav", "").split("-")
    class_id = int(parts[1])
    fold = int(os.path.basename(os.path.dirname(file_path)).replace("fold", ""))
    return class_id, fold


def parse_esc50_filename(file_path):
    """
    Parse ESC-50 filename convention:
        [fold]-[clip_id]-[take]-[target].wav
    Returns (target_class_id, fold).
    """
    file_name = os.path.basename(file_path)
    parts = file_name.replace(".wav", "").split("-")
    fold = int(parts[0])
    target = int(parts[3])
    return target, fold


def extract_embeddings(audio_dir, output_path, dataset="us8k", batch_size=64):
    """
    Extract LAION-CLAP embeddings from all .wav files under audio_dir.

    Args:
        audio_dir:   Directory with fold*/  subdirectories containing .wav files
        output_path: Path to save the .pt embedding dict
        dataset:     "us8k" or "esc50" — determines filename parsing
        batch_size:  Files processed per forward pass (default 64)
    """
    model = load_model()

    wav_files = glob.glob(os.path.join(audio_dir, "fold*", "*.wav"))
    if not wav_files:
        # Also try flat directory (no fold subdirs)
        wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found under {audio_dir}")

    print(f"Found {len(wav_files)} files. Extracting embeddings...")
    parse_fn = parse_us8k_filename if dataset == "us8k" else parse_esc50_filename

    feat_data = {}
    for i in tqdm(range(0, len(wav_files), batch_size)):
        batch_paths = wav_files[i:i + batch_size]
        try:
            with torch.no_grad():
                audio_embds = model.get_audio_embedding_from_filelist(batch_paths)
            for idx, embd in enumerate(audio_embds):
                path = batch_paths[idx]
                file_name = os.path.basename(path)
                class_id, fold = parse_fn(path)
                embd_tensor = torch.tensor(embd).float()
                embd_tensor = F.normalize(embd_tensor, p=2, dim=-1)
                feat_data[file_name] = {
                    "class_gt": class_id,
                    "fold": fold,
                    "embd": embd_tensor,
                }
        except Exception as e:
            print(f"  Error on batch {i // batch_size}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(feat_data, output_path)
    print(f"Saved {len(feat_data)} embeddings → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract LAION-CLAP embeddings from a directory of audio files."
    )
    parser.add_argument(
        "--audio_dir", required=True,
        help="Directory containing fold*/  subdirectories with .wav files"
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Output .pt file path (e.g. ./data/embeddings/us8k_noisy_street_traffic.pt)"
    )
    parser.add_argument(
        "--dataset", default="us8k", choices=["us8k", "esc50"],
        help="Dataset type for filename parsing (default: us8k)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Files per forward pass (default: 64)"
    )
    args = parser.parse_args()
    extract_embeddings(args.audio_dir, args.output_path, args.dataset, args.batch_size)


if __name__ == "__main__":
    main()
