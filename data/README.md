# Data Setup

This directory is where you place your audio datasets and embeddings.

## Folder Layout (after setup)

```
data/
├── input/
│   ├── urbansound8k/              ← raw US8K audio (fold1/ ... fold10/)
│   ├── urbansound8k_<bg>/         ← noisy US8K (one folder per TAU background)
│   ├── esc50/                     ← raw ESC-50 audio
│   ├── esc50_<bg>/                ← noisy ESC-50
│   └── tau2019uas/                ← TAU Urban Acoustic Scenes 2019 backgrounds
│       └── TAU-urban-acoustic-scenes-2019-development/audio/
│           ├── airport/
│           ├── bus/
│           ├── metro/
│           ├── metro_station/
│           ├── park/
│           ├── public_square/
│           ├── shopping_mall/
│           ├── street_pedestrian/
│           ├── street_traffic/
│           └── tram/
└── embeddings/
    ├── urbansound8k_clean.pt
    ├── urbansound8k_noisy_<bg>.pt
    ├── esc50_clean.pt
    └── esc50_noisy_<bg>.pt
```

---

## Step 1 — Download the Datasets

### UrbanSound8K
Homepage: https://urbansounddataset.weebly.com/urbansound8k.html  
Direct download: https://zenodo.org/record/1203745

```bash
# After downloading and extracting:
mv UrbanSound8K/ data/input/urbansound8k/
```

### ESC-50
Homepage: https://github.com/karolpiczak/ESC-50  
```bash
git clone https://github.com/karolpiczak/ESC-50.git data/input/esc50
```

### TAU Urban Acoustic Scenes 2019 (noise backgrounds)
Homepage: https://zenodo.org/record/2589280  
Download the **Development dataset** (TAU-urban-acoustic-scenes-2019-development).

```bash
# After downloading and extracting:
mv TAU-urban-acoustic-scenes-2019-development/ \
   data/input/tau2019uas/TAU-urban-acoustic-scenes-2019-development/
```

---

## Step 2 — Inject Noise (US8K + TAU backgrounds)

Run once per TAU background environment. Example for `street_traffic`:

```bash
python data/inject_noise.py \
    --folds 1,2,3,4,5,6,7,8,9,10 \
    --parameters "ref_db=-36 seed=123 n_soundscapes=1 duration=10.0 \
                  event_time=(truncnorm,3.0,1.5,0.0,6.0) \
                  snr_dist=(uniform,6,10) bg=street_traffic"
```

Repeat for each of the 10 backgrounds:
`airport, bus, metro, metro_station, park, public_square,`  
`shopping_mall, street_pedestrian, street_traffic, tram`

> **Requires:** `pip install scaper`

---

## Step 3 — Extract LAION-CLAP Embeddings

Run once for each audio directory (clean + 10 noisy backgrounds):

```bash
# Clean US8K
python scripts/extract_embeddings.py \
    --audio_dir data/input/urbansound8k/audio \
    --output_path data/embeddings/urbansound8k_clean.pt \
    --dataset us8k

# Noisy US8K (street_traffic background)
python scripts/extract_embeddings.py \
    --audio_dir data/input/urbansound8k_street_traffic \
    --output_path data/embeddings/urbansound8k_noisy_street_traffic.pt \
    --dataset us8k
```

> **Note:** The LAION-CLAP checkpoint (`630k-audioset-fusion-best.pt`, ~600 MB)
> is downloaded automatically on the first run via `model.load_ckpt()`.

---

## Step 4 — Run PRISM

```bash
python examples/example_us8k.py
python examples/example_esc50.py
```
