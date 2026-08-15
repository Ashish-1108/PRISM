"""PRISM on ESC-50 5-fold CV, 10 TAU backgrounds. Usage: python examples/example_esc50.py"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism import prism
from prism.prompt_ensemble import get_multi_prompt_text_features

# ── Config ──────────────────────────────────────────────────────────────────
EMBED_DIR = './data/embeddings'
CLEAN_FILE = 'esc50_clean.pt'

NOISY_FILES = {
    "airport":           "esc50_noisy_airport.pt",
    "bus":               "esc50_noisy_bus.pt",
    "metro":             "esc50_noisy_metro.pt",
    "metro_station":     "esc50_noisy_metro_station.pt",
    "park":              "esc50_noisy_park.pt",
    "public_square":     "esc50_noisy_public_square.pt",
    "shopping_mall":     "esc50_noisy_shopping_mall.pt",
    "street_pedestrian": "esc50_noisy_street_pedestrian.pt",
    "street_traffic":    "esc50_noisy_street_traffic.pt",
    "tram":              "esc50_noisy_tram.pt",
}

N_CLASSES = 50


def load_esc50(noisy_path, clean_path, bg):
    """Load ESC-50 embedding pairs — noisy filename has background suffix."""
    noisy = torch.load(noisy_path, weights_only=False, map_location='cpu')
    clean = torch.load(clean_path, weights_only=False, map_location='cpu')
    emb_n, emb_c, cls_arr, folds_arr = [], [], [], []
    for ck in sorted(clean.keys()):
        nk = ck.replace('.wav', '') + f'-{bg}-0.wav'
        if nk in noisy:
            n_e = noisy[nk]['embd']
            c_e = clean[ck]['embd']
            emb_n.append(torch.tensor(n_e).float() if not torch.is_tensor(n_e) else n_e.float())
            emb_c.append(torch.tensor(c_e).float() if not torch.is_tensor(c_e) else c_e.float())
            cls_arr.append(noisy[nk]['target'])
            folds_arr.append(noisy[nk]['fold'])
    E_n = F.normalize(torch.stack(emb_n), p=2, dim=1)
    E_c = F.normalize(torch.stack(emb_c), p=2, dim=1)
    return E_n, E_c, np.array(cls_arr), np.array(folds_arr)


def main():
    import laion_clap
    print("Loading LAION-CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt()

    # ESC-50 class map: 0..49
    # (full 50-class list; fill in class names from ESC-50 metadata if desired)
    esc50_classes = {i: f"class_{i}" for i in range(N_CLASSES)}
    T = get_multi_prompt_text_features(esc50_classes, model)
    del model

    clean_path = os.path.join(EMBED_DIR, CLEAN_FILE)

    print(f"\n{'Background':<20s}  {'Zero-Shot':>10s}  {'PRISM':>10s}  {'Δ':>8s}")
    print("─" * 56)

    all_noisy, all_prism = [], []

    for bg, fname in NOISY_FILES.items():
        noisy_path = os.path.join(EMBED_DIR, fname)
        if not os.path.exists(noisy_path):
            print(f"  [skip] {fname} not found")
            continue

        E_n, _, classes, folds = load_esc50(noisy_path, clean_path, bg)

        noisy_accs, prism_accs = [], []
        for fold in sorted(set(folds.tolist())):
            test_mask = folds == fold
            train_mask = ~test_mask
            gt = classes[test_mask]

            preds_noisy = (E_n[test_mask] @ T.t()).argmax(dim=1).numpy()
            noisy_accs.append(accuracy_score(gt, preds_noisy) * 100)

            E_all = torch.cat([E_n[train_mask], E_n[test_mask]])
            n_train = int(train_mask.sum())
            E_denoised, _ = prism(E_all, T, N_CLASSES)
            E_test = E_denoised[n_train:]

            preds_prism = (E_test @ T.t()).argmax(dim=1).numpy()
            prism_accs.append(accuracy_score(gt, preds_prism) * 100)

        noisy_avg = np.mean(noisy_accs)
        prism_avg = np.mean(prism_accs)
        all_noisy.append(noisy_avg)
        all_prism.append(prism_avg)
        print(f"{bg:<20s}  {noisy_avg:9.2f}%  {prism_avg:9.2f}%  {prism_avg - noisy_avg:+7.2f}%")

    if all_noisy:
        print("─" * 56)
        print(f"{'Average':<20s}  {np.mean(all_noisy):9.2f}%  {np.mean(all_prism):9.2f}%  "
              f"{np.mean(all_prism) - np.mean(all_noisy):+7.2f}%")


if __name__ == '__main__':
    main()
