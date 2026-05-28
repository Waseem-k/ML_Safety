# ML Safety — CARLA Object Detection

Practical exercises for the *Introduction to Machine Learning Safety* course.  
Binary classifiers (pedestrian · traffic light · vehicle) trained on a CARLA driving-simulator dataset, plus solutions to Exercise Sheet 5 (Testing LLMs & Agents).

---

## Repository layout

```
ML Safety/
├── 2026/                          # CARLA dataset
│   ├── train/train/
│   │   ├── rgb-front/             # JPG frames (~500×500 px)
│   │   ├── segmentation-front/
│   │   └── labels.csv
│   ├── validation/validation/     # same structure
│   ├── test/test/                 # standard test split
│   ├── test-fog/test-fog/         # domain-shift: fog
│   ├── test-night/test-night/     # domain-shift: night
│   └── test-town-01/test-town-01/ # domain-shift: unseen town
│
├── notebooks/
│   ├── approach1_resnet18/
│   │   ├── 01_pedestrian_resnet18.ipynb
│   │   ├── 02_traffic_light_resnet18.ipynb
│   │   └── 03_vehicle_resnet18.ipynb
│   ├── approach2_efficientnet/
│   │   ├── 04_pedestrian_efficientnet.ipynb
│   │   ├── 05_traffic_light_efficientnet.ipynb
│   │   └── 06_vehicle_efficientnet.ipynb
│   └── 07_solutions_exercise5.ipynb
│
├── 00_overview.pdf
├── 020_system_safety.pdf
├── 020_Exercise_system_safety.pdf
├── 025_fundamentals.pdf
├── 030_testing.pdf
└── 032_testing-llms.pdf
```

---

## Dataset

Recorded with the [CARLA](https://carla.org/) simulator. Each split contains front-facing RGB frames and a `labels.csv` with per-frame binary labels.

| Column | Type | Description |
|--------|------|-------------|
| `frame` | string | 6-digit zero-padded frame ID (e.g. `000010`) |
| `has_pedestrian` | bool | Pedestrian visible in frame |
| `has_traffic_light` | bool | Traffic light visible in frame |
| `has_vehicle` | bool | Vehicle visible in frame |
| `px_pedestrian` | int | Pixel count of pedestrian |
| `px_traffic_light` | int | Pixel count of traffic light |
| `px_vehicle` | int | Pixel count of vehicle |

**Class imbalance**: `has_vehicle` is near-always `True`. `has_pedestrian` and `has_traffic_light` are minority-positive.

**Domain-shift test sets**: `test-fog`, `test-night`, and `test-town-01` are held-out splits for out-of-distribution evaluation.

---

## Notebooks

### Approach 1 — ResNet-18 (notebooks 01–03)

Baseline fine-tuning approach. One notebook per detection task (pedestrian / traffic light / vehicle), all sharing the same architecture and training recipe.

| Technique | Detail |
|-----------|--------|
| Backbone | ResNet-18, ImageNet pretrained (`torchvision`) |
| Loss | `BCEWithLogitsLoss` with auto-computed `pos_weight` |
| Optimiser | Adam + CosineAnnealingLR |
| Training | 2-phase: frozen backbone (5 ep) → full fine-tune (20 ep) |
| Precision | bfloat16 autocast on A100 |
| Outputs | Training curves · Confusion matrix · ROC curve · `results.json` |

### Approach 2 — EfficientNet-B3 (notebooks 04–06)

Advanced training approach with additional techniques to handle class imbalance and improve generalisation.

| Technique | Detail |
|-----------|--------|
| Backbone | EfficientNet-B3 (`timm`), ImageNet pretrained |
| Loss | Focal Loss (γ=2.0, α=0.25; α=0.75 for Vehicle) |
| Sampler | `WeightedRandomSampler` for balanced mini-batches |
| Augmentation | MixUp (α=0.4) + RandAugment(ops=2, mag=9) + RandomErasing |
| Optimiser | AdamW + OneCycleLR (10% warmup) |
| Unfreezing | Progressive: head-only (ep 1–3) → 4 stages (ep 4–7) → full (ep 8–10) |
| Explainability | GradCAM on `conv_head` |
| Inference | 6-pass Test-Time Augmentation (TTA) |
| Outputs | Training curves · GradCAM overlays · Confusion matrix · ROC · `results.json` |

### Exercise Sheet 5 Solutions (notebook 07)

Solutions to all exercises in `032_testing-llms.pdf`.

| Exercise | Content |
|----------|---------|
| 5.1 | Human pairwise evaluation design (Bradley-Terry, Elo, LLM-judge bias mitigations) |
| 5.2 | Coding-agent evaluation: trajectory quality, prompt injection via README |
| 5.3 | Training-data poisoning: backdoor mechanics, why 250 samples is alarming, safeguards |
| 5.4 | Temperature scaling on the pedestrian detector; safety constraint analysis (θ=0.6) |
| 5.5 | Backdoor attack: 10×10 red-square trigger, 10% label-flip poisoning, ASR measurement |

---

## Setup

### Requirements

```
torch >= 2.0
torchvision
timm
scikit-learn
matplotlib
seaborn
Pillow
pandas
numpy
```

### Google Colab Pro (recommended)

All notebooks are written for **Colab Pro with an A100 GPU** and assume the dataset is stored in Google Drive.

**1. Upload dataset to Drive**

Place the contents of the `2026/` folder at:

```
MyDrive/ML_Safety_2026/train/train/
MyDrive/ML_Safety_2026/validation/validation/
MyDrive/ML_Safety_2026/test/test/
MyDrive/ML_Safety_2026/test-fog/test-fog/
MyDrive/ML_Safety_2026/test-night/test-night/
MyDrive/ML_Safety_2026/test-town-01/test-town-01/
```

**2. Open a notebook**

Open any `.ipynb` from `notebooks/` via VSCode connected to a Colab runtime, or upload directly to Colab.

**3. Run all cells top-to-bottom**

The second cell after configuration copies the dataset from Drive to the Colab local SSD (`/content/ML_Safety_2026`). This takes ~5–10 min once per session but makes each training epoch **3–5× faster** than reading directly from Drive.

Checkpoints are saved back to Drive at:

```
MyDrive/ML_Safety_2026/checkpoints/<Task>/<Backbone>/best_model.pth
```

**4. Connecting VSCode to a Colab runtime**

After starting a runtime in the Colab browser tab:

1. `Ctrl+Shift+P` → **Jupyter: Specify Jupyter Server for Connections**
2. Choose **Existing**
3. Paste the runtime URL (get it from the Colab tab — it changes each session)

---

## Expected training time (A100, dataset on local SSD)

| Approach | Epochs | Est. time per notebook |
|----------|--------|------------------------|
| ResNet-18 | 25 total (5 frozen + 20 fine-tune) | ~25–40 min |
| EfficientNet-B3 | 10 | ~30–50 min |

---

## Checkpoint layout (after training)

```
MyDrive/ML_Safety_2026/checkpoints/
├── Pedestrian/
│   ├── ResNet18/best_model.pth
│   └── EfficientNetB3/best_model.pth
├── TrafficLight/
│   ├── ResNet18/best_model.pth
│   └── EfficientNetB3/best_model.pth
└── Vehicle/
    ├── ResNet18/best_model.pth
    └── EfficientNetB3/best_model.pth
```

The Exercise 5 notebook (`07_solutions_exercise5.ipynb`) loads the pedestrian ResNet-18 checkpoint for temperature scaling and backdoor experiments. Run notebooks 01–06 first.

---

## Course materials

| File | Contents |
|------|----------|
| `00_overview.pdf` | Course overview |
| `020_system_safety.pdf` | Lecture: system safety |
| `020_Exercise_system_safety.pdf` | Exercise sheet: system safety |
| `025_fundamentals.pdf` | Lecture: ML fundamentals (source of the 3-classifier task) |
| `030_testing.pdf` | Lecture: testing ML systems |
| `032_testing-llms.pdf` | Exercise sheet 5: testing LLMs & agents (solved in notebook 07) |
