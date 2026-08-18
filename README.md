<div align="center">
# PRISM: Prototype-Rectified Iterative Self-supervised Manifold Denoising

<a href="ASHISH_PROFILE_URL">Ashish Anand Shukla</a> ,
<a href="RINI_PROFILE_URL">Rini Smita Thakur</a> ,
<a href="aryan-das.netlify.app">Aryan Das</a> ,
<a href="VINOD_PROFILE_URL">Vinod K. Kurmi</a>


Official implementation of the CIKM 2026 paper:  
**"Prototype-Rectified Iterative Self-supervised Manifold Denoising under Severe Acoustic Shift"**
</div>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2608.15037-b31b1b.svg)](https://arxiv.org/abs/2608.15037)

Developed at the **VisDom Lab (Visual Data Computing Group)**, Department of Data Science and Engineering (DSE), Indian Institute of Science Education and Research (IISER) Bhopal.

![PRISM Architecture](figures/fig_architecture_prism-1.png)

---

## Setup

**Clone and install**
```bash
git clone https://github.com/Ashish-1108/PRISM.git
cd PRISM
pip install -r requirements.txt
```

**Download the LAION-CLAP model**
```bash
mkdir -p models
wget -P models https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt
```

**Download datasets**

See [data/README.md](data/README.md) for download links (US8K, ESC-50, TAU 2019).

---

## Create Noisy Soundscapes

Mix foreground audio with TAU Urban Acoustic Scene backgrounds using [Scaper](https://scaper.readthedocs.io/):

```bash
python data/inject_noise.py \
    --folds 1,2,3,4,5,6,7,8,9,10 \
    --parameters "ref_db=-36 seed=123 snr_dist=(uniform,6,10) bg=street_traffic"
```

Repeat for each background: `airport, bus, metro, metro_station, park, public_square, shopping_mall, street_pedestrian, street_traffic, tram`

---

## Extract Audio Embeddings

```bash
# Clean audio
python scripts/extract_embeddings.py \
    --audio_dir data/input/urbansound8k/audio \
    --output_path data/embeddings/urbansound8k_clean.pt \
    --dataset us8k

# Noisy audio (one per background)
python scripts/extract_embeddings.py \
    --audio_dir data/input/urbansound8k_street_traffic \
    --output_path data/embeddings/urbansound8k_noisy_street_traffic.pt \
    --dataset us8k
```

---

## Classify Audio Files (Direct Usage)

Run PRISM directly on a folder of `.wav` files — no pre-computed embeddings needed:

```bash
python scripts/classify.py \
    --audio_folder path/to/your/audio \
    --class_labels demo/us8k_labels.txt
```

---

## Reproduce Paper Results

```bash
python examples/example_us8k.py    # UrbanSound8K
python examples/example_esc50.py   # ESC-50
```

---

## Quick API Usage

```python
from prism import prism, deploy
from prism.prompt_ensemble import get_multi_prompt_text_features

# Build text prototypes (once)
label_map = {0: 'air_conditioner', 1: 'car_horn', ..., 9: 'street_music'}
T = get_multi_prompt_text_features(label_map, clap_model)

# Calibrate on a batch (no labels needed)
E_denoised, W = prism(E_noisy, T, n_classes=10)

# Deploy on new samples instantly (~0.0009 ms/sample)
E_clean = deploy(E_new_sample, W)
prediction = (E_clean @ T.t()).argmax(dim=1)
```

---

## Citation

```bibtex
@inproceedings{shukla2026prism,
  author    = {Shukla, Ashish Anand and Thakur, Rini Smita and Das, Aryan and Kurmi, Vinod K.},
  title     = {Prototype-Rectified Iterative Self-supervised Manifold Denoising under Severe Acoustic Shift},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  month     = {November},
  address   = {Rome, Italy},
  publisher = {ACM},
  doi       = {10.1145/3799682.3840694},
  eprint    = {2608.15037},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD}
}
```

---
*Note: The DOI will resolve in the ACM Digital Library around the conference date (November 2026).*

Preprint also available on arXiv: [arXiv:2608.15037](https://arxiv.org/abs/2608.15037)

Noise injection pipeline adapted from [AudioText-ContextDA](https://github.com/eacevedo1/AudioText-ContextDA) (Apache-2.0).
