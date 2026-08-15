"""PRISM on UrbanSound8K 10-fold CV, 10 TAU backgrounds. Usage: python examples/example_us8k.py"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism import prism, deploy
from prism.prompt_ensemble import get_multi_prompt_text_features

# Update these paths to your local embedding files
EMBED_DIR = './data/embeddings'
CLEAN_FILE = 'urbansound8k_clean.pt'

NOISY_FILES = {
    "airport":           "urbansound8k_noisy_airport.pt",
    "bus":               "urbansound8k_noisy_bus.pt",
    "metro":             "urbansound8k_noisy_metro.pt",
    "metro_station":     "urbansound8k_noisy_metro_station.pt",
    "park":              "urbansound8k_noisy_park.pt",
    "public_square":     "urbansound8k_noisy_public_square.pt",
    "shopping_mall":     "urbansound8k_noisy_shopping_mall.pt",
    "street_pedestrian": "urbansound8k_noisy_street_pedestrian.pt",
    "street_traffic":    "urbansound8k_noisy_street_traffic.pt",
    "tram":              "urbansound8k_noisy_tram.pt",
}

US8K_CLASSES = {
    0: 'air_conditioner', 1: 'car_horn', 2: 'children_playing',
    3: 'dog_bark', 4: 'drilling', 5: 'engine_idling',
    6: 'gun_shot', 7: 'jackhammer', 8: 'siren', 9: 'street_music'
}
N_CLASSES = 10


def load_embeddings(noisy_path, clean_path):
    """Load and align noisy/clean embedding pairs by filename."""
    noisy = torch.load(noisy_path, weights_only=False, map_location='cpu')
    clean = torch.load(clean_path, weights_only=False, map_location='cpu')
    emb_n, emb_c, cls_arr, folds_arr = [], [], [], []
    for k in noisy:
        base = k.replace('.wav', '').rsplit('-', 2)[0] + '.wav'
        if base in clean:
            n_e = noisy[k]['embd']
            c_e = clean[base]['embd']
            emb_n.append(torch.tensor(n_e).float() if not torch.is_tensor(n_e) else n_e.float())
            emb_c.append(torch.tensor(c_e).float() if not torch.is_tensor(c_e) else c_e.float())
            cls_arr.append(noisy[k]['class_gt'])
            folds_arr.append(noisy[k]['fold'])
    E_n = F.normalize(torch.stack(emb_n), p=2, dim=1)
    E_c = F.normalize(torch.stack(emb_c), p=2, dim=1)
    return E_n, E_c, np.array(cls_arr), np.array(folds_arr)


def get_text_prototypes(model):
    """
    Build text prototypes via 20-prompt ensemble averaging.
    Uses get_multi_prompt_text_features() — same function used in the paper.
    """
    return get_multi_prompt_text_features(US8K_CLASSES, model)


def main():
    # Load LAION-CLAP for text prototypes
    import laion_clap
    print("Loading LAION-CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt()  # downloads 630k-audioset-fusion-best.pt on first run
    T = get_text_prototypes(model)
    del model  # free GPU memory before evaluation

    clean_path = os.path.join(EMBED_DIR, CLEAN_FILE)

    print(f"\n{'Background':<20s}  {'Zero-Shot':>10s}  {'PRISM':>10s}  {'Δ':>8s}")
    print("─" * 56)

    all_noisy, all_prism = [], []

    for bg, fname in NOISY_FILES.items():
        noisy_path = os.path.join(EMBED_DIR, fname)
        if not os.path.exists(noisy_path):
            print(f"  [skip] {fname} not found")
            continue

        E_n, _, classes, folds = load_embeddings(noisy_path, clean_path)

        noisy_accs, prism_accs = [], []
        for fold in sorted(set(folds.tolist())):
            test_mask = folds == fold
            train_mask = ~test_mask
            gt = classes[test_mask]

            # Noisy zero-shot baseline
            preds_noisy = (E_n[test_mask] @ T.t()).argmax(dim=1).numpy()
            noisy_accs.append(accuracy_score(gt, preds_noisy) * 100)

            # PRISM: calibrate transductively on train+test, evaluate on test
            E_all = torch.cat([E_n[train_mask], E_n[test_mask]])
            n_train = int(train_mask.sum())

            E_denoised, W = prism(E_all, T, N_CLASSES)
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

    print("\nTo deploy PRISM on a new audio sample after calibration:")
    print("  E_clean = deploy(E_new_noisy, W)")
    print("  pred    = (E_clean @ T.t()).argmax(dim=1)")


if __name__ == '__main__':
    main()
