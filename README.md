# ML Safety — CARLA Perception Safety Case

Practical exercises for the *Introduction to Machine Learning Safety* course (Otto-von-Guericke University Magdeburg). Three binary classifiers (pedestrian · traffic light · vehicle) are trained on a CARLA driving-simulator dataset and then subjected to the full course pipeline: testing/ODD coverage, uncertainty calibration, adversarial robustness, and OOD detection. The results are the evidence base for `Final_Report.tex` and `STPA_Consolidated.md`.

> **Note:** this README describes what is actually in this repository and was actually run to produce `results/`. An earlier draft of this file described a Colab-notebook / EfficientNet-B3 workflow; that workflow's notebooks are not present here and were not the source of any number in the report. Everything below is grounded in the checked-in `.py` scripts and their logged output.

---

## 1. Repository layout

```
ML_Safety/
├── ml_safety_ex_3.py                  # Exercise 3: trains the 3 ResNet-18 classifiers
├── ex4_testing.py                     # Exercise 4: recall/precision/F1 + k-projection ODD coverage
├── ex7_uncertainty_calibration.py     # Exercise 7: ECE, temperature scaling, cost-optimal threshold
├── ex8_fgsm_attack.py                 # Exercise 8: FGSM adversarial robustness
├── ex9_msp_baseline.py                # Exercise 9.6: MSP out-of-distribution baseline
├── ex9_feature_ood.py                 # Exercise 9.7: Mahalanobis feature-space OOD detector
├── ex9_visualise_shift.py             # Exercise 9.4: sample images + confidence under shift
├── Excercise_6_IMLS.ipynb             # Exercise 6: Grad-CAM explainability (see §5.3)
│
├── odd_coverage/                      # Vendored k-projection coverage implementation
│   ├── __init__.py                    #   (Cheng et al., 2018; used unmodified, see file header)
│   ├── kprojection.py
│   ├── LICENSE                        #   upstream license for the vendored code
│   ├── Control_Structure_Diagram.png  # Exercise 2: hand-drawn control-structure diagram
│   └── ODD_Coverage_extended.png      # Extended control structure (report Fig., Control structure)
│
├── figures/                            # Figures used by the report's \includegraphics calls
│   ├── gradcam/                        #   Full-resolution Grad-CAM originals (5 images, Ex. 6)
│   ├── gradcam_*_cropped.png           #   Cropped Grad-CAM overlays used in report §5 V-1 (×4)
│   ├── forgery_original.jpg            #   Exploratory probe: original frame
│   ├── forgery_doctored.png            #   Exploratory probe: generative edit
│   └── ODD_Coverage_extended.png       #   Control-structure diagram used in the report
│
├── models/                            # Checkpoints (.pth) + all generated figures (.png)
├── results/                           # JSON metrics + full stdout logs (.txt) per exercise
│
├── Exercise_8_Answers.md              # Written answers, Exercise 8 (theory + Ex. 8.6 STPA extension)
├── Exercise_9_Answers.md              # Written answers, Exercise 9 (theory + Ex. 9.8 STPA extension)
├── STPA_Consolidated.md               # Merged STPA tables (Ex. 2 base + Ex. 8.6 + Ex. 9.8 + Ex. 7.7)
├── report_system_description_odd.md   # Draft: report Sections 2–3 (superseded by Final_Report.tex)
├── requirements.txt
├── LICENSE                            # Repository license
├── .gitignore
└── README.md                          # this file
```

---

## 2. Dataset

Recorded with the [CARLA](https://carla.org/) simulator. Not nested inside `ML_Safety/` — every script expects a dataset directory (default `2026`, override with `--data-dir`) containing `train\`, `validation\`, `test\`, and three domain-shift splits (`test-fog\`, `test-night\`, `test-town-01\`), each with an `rgb-front\` image folder plus `labels.csv`, `actions.feather`, and `weather.feather`.

| Column | Type | Description |
|--------|------|-------------|
| `frame` | int | Original CARLA simulation tick, zero-padded to 6 digits for the image filename (e.g. `000010.jpg`) |
| `has_pedestrian` | bool | Pedestrian visible in frame |
| `has_traffic_light` | bool | Traffic light visible in frame |
| `has_vehicle` | bool | Vehicle visible in frame |
| `px_pedestrian` / `px_traffic_light` / `px_vehicle` | int | Pixel count of each object (used to bin object size for ODD coverage) |

`train/train/actions.feather` additionally holds `steer`, `throttle`, `brake` per frame, **positionally** aligned with `labels.csv` (its own `frame` column is just a row index, not the CARLA tick — see the note in `ex4_testing.py::build_scenarios`).

**Class imbalance** (in-distribution test split, 3,600 images): `has_pedestrian` 706/3,600 positive (~20%), `has_traffic_light` 2,584/3,600 (~72%), `has_vehicle` 2,700/3,600 (~75%). `has_pedestrian` is the minority-positive, safety-critical class. (On the 7,200-image *training* split the pedestrian positive rate is ~24%, 1,718/7,200 — see `results/ex9_feature_run_log.txt`.)

**Domain-shift test sets**: `test-fog`, `test-night`, `test-town-01` are held-out splits used only for OOD evaluation (Exercise 9); they are never used for training.

**Verifying the ODD gap claim (report §3.1).** `train` and `test` are not just named as in-distribution — their `weather.feather` values are literally constant (`min == max` on every field: sun altitude 45°, fog density 2, precipitation 0), recorded in `Town02`. The domain-shift splits use categorically different presets, confirmed the same way: `test-fog` has fog density 80 (40× train), `test-night` has sun altitude −90° with precipitation 70, and `test-town-01` is recorded in `Town01`, a map that never appears in `train`. Reproduce this check with:

```python
import pandas as pd
for split in ["train/train", "test/test", "test-fog/test-fog", "test-night/test-night", "test-town-01/test-town-01"]:
    print(split, pd.read_feather(f"{split}/weather.feather").describe().loc[["min", "max"]])
```

If your dataset lives elsewhere, every script accepts `--data-dir <path>` pointing at the folder that contains `train/`, `validation/`, `test/`, etc. (i.e. pass the `2026` folder itself, not its parent).

---

## 3. Setup

```bash
python -m venv venv
venv/Scripts/activate            # Windows; use `source venv/bin/activate` on Unix
pip install -r requirements.txt
pip install pyarrow              # needed for pd.read_feather in ex4_testing.py — not pinned in requirements.txt
```

`requirements.txt`:
```
torch>=2.4.0
torchvision>=0.19.0
tqdm>=4.66.0
seaborn>=0.13.0
numpy==2.3.1
pandas==2.3.0
matplotlib==3.10.9
Pillow==12.2.0
scikit-learn>=1.5.0
```

All scripts auto-detect CUDA (`torch.cuda.is_available()`) and fall back to CPU; the logs in `results/` were captured on CPU (`Device : cpu`), so expect a GPU run to be substantially faster.

---

## 4. Model

**One architecture, one training recipe, three independently trained checkpoints** — there is no second (EfficientNet-B3) approach in this repository; only the ResNet-18 classifiers below back every number in the report.

| Element | Configuration |
|---|---|
| Backbone | `torchvision.models.resnet18`, ImageNet-pretrained, final FC replaced with `nn.Linear(512, 2)` |
| Loss | `nn.CrossEntropyLoss()` (2-way softmax, not `BCEWithLogitsLoss`) |
| Optimizer | Adam, learning rate `1e-4` |
| Training regime | Single phase, full fine-tuning (no frozen-backbone phase) |
| Epochs | 5 (script default; confirmed by the three manually-saved loss curves — see §5.1) |
| Batch size | 16 |
| Train split | `train/train`, 7,200 images, one binary target column per run |
| Feature representation | 512-d `avgpool` activation, extracted via forward hook for Mahalanobis OOD scoring (Exercise 9.7) |

All three tasks (`has_pedestrian`, `has_traffic_light`, `has_vehicle`) share this exact recipe; only `--target-label`-equivalent selection differs. Checkpoints are saved as `resnet18_<label>.pth` and every downstream script (`ex4`, `ex7`, `ex8`, `ex9_*`) loads them from `models/` by that exact naming convention.

---

## 5. Reproducing every result

Run from `ML_Safety/` with the venv active. All commands below use each script's built-in `--data-dir` default / `--models-dir models`; override both with `--data-dir <path>` / `--models-dir <path>` if your paths differ.

Subsections below follow the order evidence appears in the report (§3 Architecture & Training → §3.2 ODD Coverage / §5 V-1 → §5 V-2 → §5 V-3 → §5 V-4 → Additional Material). **Exercise 5** (`032_testing-llms.pdf`, testing LLMs & agents) has no corresponding script or answer file in this repository — it's out of scope for this CARLA perception safety case and is not reproduced here (see §7).

### 5.1 Train the three classifiers (Exercise 3)

```bash
python ml_safety_ex_3.py --data-dir <path-to-2026> --epochs 5
```

Saves `resnet18_has_pedestrian.pth`, `resnet18_has_traffic_light.pth`, `resnet18_has_vehicle.pth` to `--save-dir` (defaults to `--data-dir`; move/copy them into `ML_Safety\models\` before running any exercise below, since that is where `ex4`/`ex7`/`ex8`/`ex9_*` look by default). Pass `--skip-eda` to suppress the class-distribution and example-image plots.

`train_binary_model()` plots the per-epoch loss curve with `plt.show()` only — unlike every other exercise script, it does **not** call `plt.savefig()`, so the curve is not persisted automatically. The three curves actually used as evidence for the 5-epoch training regime (report §3, Architecture & Training) were saved manually to the `ML_Safety/` root: `Pedestrian_training_graph.png`, `traffic_training_graph.png`, `training curve for has vehicle.png`. Each plots epochs 1–5 with monotonically decreasing cross-entropy loss, and each file's timestamp immediately precedes the corresponding `.pth` checkpoint's save timestamp (pedestrian 12:30, traffic light 17:15–17:16, vehicle 20:10–20:11, all Jun 4), confirming they come from the actual runs that produced the deployed checkpoints.

### 5.2 Exercise 4 — testing & ODD coverage → `results/ex4_run_log.txt`

```bash
python ex4_testing.py > results/ex4_run_log.txt 2>&1
```

Produces:
- `results/ex4_per_class_metrics.json`, `results/ex4_kprojection_coverage.json`
- `models/ex4_confusion_<label>.png` (×3)
- Per-class precision/recall/F1/confusion matrix (report §5, V-1) and the $k$-projection ODD coverage table for $k=1,2,3$ (report §3.2), computed over 7 dimensions derived from `labels.csv` + `actions.feather` (presence flags, object-size bins, steering regime, driving-dynamics regime — see the docstring in `build_scenarios()` for why weather/lighting are intentionally excluded from the projection).

### 5.3 Exercise 6 — Grad-CAM explainability → report §5, V-1 "Supporting Evidence: Explanation Quality"

Grad-CAM (last conv layer, target = positive class) was run inside `Excercise_6_IMLS.ipynb` (Approach 2's notebook pipeline, on `conv_head`), not from a standalone script. The notebook's `ROOT` path (`D:\Dev\IMLS Exercises`) does not exist on this machine, so the overlay PNGs it saves under `Exercise_6/figures/` were never written to disk here — but the notebook's own cell outputs (from the run that originally produced them) still had the 5 images embedded as base64 `image/png` data. Extracted with:

```bash
python -c "
import json, base64
nb = json.load(open('Excercise_6_IMLS.ipynb', encoding='utf-8'))
names = ['gradcam_correct.png', 'gradcam_misclassified.png', 'gradcam_test-fog.png', 'gradcam_test-night.png', 'gradcam_test-town-01.png']
idx = 0
for cell in nb['cells']:
    for out in cell.get('outputs', []):
        data = out.get('data', {})
        if 'image/png' in data:
            b64 = data['image/png']
            b64 = ''.join(b64) if isinstance(b64, list) else b64
            open(f'models/gradcam/{names[idx]}', 'wb').write(base64.b64decode(b64))
            idx += 1
"
```

No model re-run was needed — these are the exact overlays the notebook produced, just recovered from its saved outputs rather than the (missing) figures directory. The four used in the report (`gradcam_correct.png`, `gradcam_test-fog.png`, `gradcam_test-night.png`, `gradcam_test-town-01.png`) were manually cropped down to one representative frame each and saved alongside the originals in `models/gradcam/` as `*_cropped.png`, then copied into `figures/` (flat, matching the report's `\includegraphics{figures/...}` convention) as `gradcam_correct_cropped.png`, `gradcam_test-fog_cropped.png`, `gradcam_test-night_cropped.png`, `gradcam_test-town-01_cropped.png`. The fifth extracted image, `gradcam_misclassified.png`, is kept in `models/gradcam/` but is not used in the report.

### 5.4 Exercise 8 — FGSM adversarial robustness → `results/ex8_run_log.txt`, `results/ex8_vehicle_remainder_log.txt`, `results/ex8_traffic_light_log.txt`

The three logs in this repo were captured as **three separate runs**, not one call over all labels — reproduce them exactly as run:

```bash
python ex8_fgsm_attack.py --labels has_pedestrian    --epsilons 0.01,0.05      > results/ex8_run_log.txt 2>&1
python ex8_fgsm_attack.py --labels has_vehicle       --epsilons 0.05,0.10      > results/ex8_vehicle_remainder_log.txt 2>&1
python ex8_fgsm_attack.py --labels has_traffic_light --epsilons 0.01,0.05,0.10 > results/ex8_traffic_light_log.txt 2>&1
```

> **Note (Windows only):** `ex8_fgsm_attack.py` prints the `ε` symbol, which crashes with `UnicodeEncodeError` on the default `cp1252` console/file encoding once you redirect stdout to a file. Set `PYTHONIOENCODING=utf-8` before running (e.g. `set PYTHONIOENCODING=utf-8` in cmd, `$env:PYTHONIOENCODING="utf-8"` in PowerShell, or prefix the command in bash) — otherwise the script silently stops right after printing the section header and before printing any recall numbers, leaving a plot on disk but no log.

Each run saves `models/ex8_fgsm_<label>.png` (clean vs. perturbed example grid) and `models/ex8_recall_drop.png` (recall bar chart; re-running overwrites it, so run all three labels back-to-back before checking the chart, or rename the file between runs if you want to keep all three).

Results (recall on the positive class, in-distribution test set, 3,600 images):

| Model | Clean recall | ε=0.01 | ε=0.05 | ε=0.10 |
|---|---|---|---|---|
| `has_pedestrian` | 0.4703 | — | 0.0014 (−0.4688) | — |
| `has_vehicle` | 0.9044 | — | 0.0948 (−0.8096) | 0.1830 (−0.7215) |
| `has_traffic_light` | 0.9807 | 0.2825 (−0.6981) | 0.0039 (−0.9768) | 0.0004 (−0.9803) |

### 5.5 Exercise 7 — calibration & cost-optimal threshold → `results/ex7_run_log.txt`

```bash
python ex7_uncertainty_calibration.py > results/ex7_run_log.txt 2>&1
```

Produces `models/ex7_reliability_diagrams.png` and, per model: ECE before/after temperature scaling (grid search $T \in \{0.5, \dots, 3.0\}$ minimising validation NLL), the pedestrian-only cost-optimal 2×2 decision table ($C_{FN}=100$, $C_{FP}=1 \Rightarrow \tau^\ast \approx 0.0099$), and the printed Exercise 7.7 STPA extension (SC-M1/SC-S1/SC-S2). Use `--skip-theory` to suppress the printed 7.1–7.3 theory answers.

### 5.6 Exercise 9 — distribution shift & OOD detection

`ex9_visualise_shift.py` operates on the **pedestrian** detector only (`TARGET_LABEL` is hard-coded, no CLI flag — it's an exploratory/plotting script, not part of the V-4 evidence). `ex9_msp_baseline.py` and `ex9_feature_ood.py` both take a `--target-label` flag and were run for all three detectors to back V-4's full-coverage evidence table in the report.

```bash
python ex9_visualise_shift.py > results/ex9_visualise_run_log.txt 2>&1   # Ex. 9.4 — sample images + confidence per split (pedestrian only)

# Ex. 9.6 — MSP baseline AUROC, all three detectors
python ex9_msp_baseline.py --target-label has_pedestrian    > results/ex9_msp_run_log.txt 2>&1
python ex9_msp_baseline.py --target-label has_traffic_light > results/ex9_msp_run_log_has_traffic_light.txt 2>&1
python ex9_msp_baseline.py --target-label has_vehicle       > results/ex9_msp_run_log_has_vehicle.txt 2>&1

# Ex. 9.7 — Mahalanobis AUROC, all three detectors (each fits on that detector's train features)
python ex9_feature_ood.py --target-label has_pedestrian    > results/ex9_feature_run_log.txt 2>&1
python ex9_feature_ood.py --target-label has_traffic_light > results/ex9_feature_run_log_has_traffic_light.txt 2>&1
python ex9_feature_ood.py --target-label has_vehicle       > results/ex9_feature_run_log_has_vehicle.txt 2>&1
```

`--target-label` defaults to `has_pedestrian`, so the pedestrian invocation above is the same as running with no flag; that's why its log/plot filenames carry no suffix while the other two detectors' outputs are suffixed `_has_traffic_light` / `_has_vehicle`.

Produces `models/ex9_sample_images.png`, `models/ex9_confidence_per_split.png`, and per detector: `models/ex9_msp_distributions[_<label>].png`, `models/ex9_msp_roc_curves[_<label>].png`, `models/ex9_feature_distributions[_<label>].png`, `models/ex9_auroc_comparison[_<label>].png`, `models/ex9_roc_comparison[_<label>].png`.

AUROC tables quoted in `STPA_Consolidated.md` and report §5, V-4:

| Model | MSP Fog | MSP Night | MSP Town01 | Mahalanobis Fog | Mahalanobis Night | Mahalanobis Town01 |
|---|---|---|---|---|---|---|
| `has_pedestrian` | 0.4932 | 0.6367 | 0.5512 | 0.9822 | 0.9995 | 0.7086 |
| `has_traffic_light` | 0.8741 | 0.6952 | 0.8249 | 0.9973 | 1.0000 | 0.9131 |
| `has_vehicle` | 0.7247 | 0.7311 | 0.6244 | 0.9632 | 0.9934 | 0.7005 |

Mahalanobis clears the SC-4 ≥0.95 bar on fog/night for all three models; Town01 falls short for all three (excused per report §4 footnote — Town01 is in-ODD, so this is a training-coverage gap, not an OOD-monitor failure).

### 5.7 Exploratory probe: generative content removal → report, Additional Material, "Exploratory Probe: Generative Content Removal"

Not part of the safety case (single hand-crafted example, no SC verified) — an in-distribution frame was edited with a generative image model to remove a cyclist, then both the original and edited frame were scored by the actual `has_pedestrian`/`has_vehicle` checkpoints (no retraining). The pair used in the report is `figures/forgery_original.jpg` and `figures/forgery_doctored.png`.

---

## 6. Result → report cross-reference

| Script | Log | Report section(s) |
|---|---|---|
| `ex4_testing.py` | `results/ex4_run_log.txt` | §3.2 ODD Coverage; §5 V-1 |
| `Excercise_6_IMLS.ipynb` (Grad-CAM, all 3 models) | embedded notebook outputs → `models/gradcam/*.png`, cropped copies in `figures/*_cropped.png` | §5 V-1 (Explanation Quality) |
| `ex7_uncertainty_calibration.py` | `results/ex7_run_log.txt` | §5 V-3; §4 LS-6/SC-8/SC-9 |
| `ex8_fgsm_attack.py` (pedestrian) | `results/ex8_run_log.txt` | §5 V-2; §4 LS-5/SC-6 |
| `ex8_fgsm_attack.py` (vehicle) | `results/ex8_vehicle_remainder_log.txt` | §5 V-2 |
| `ex8_fgsm_attack.py` (traffic light) | `results/ex8_traffic_light_log.txt` | §5 V-2 |
| `ex9_msp_baseline.py` (pedestrian) | `results/ex9_msp_run_log.txt` | §5 V-4; §4 LS-4/SC-4 |
| `ex9_msp_baseline.py` (traffic light) | `results/ex9_msp_run_log_has_traffic_light.txt` | §5 V-4 |
| `ex9_msp_baseline.py` (vehicle) | `results/ex9_msp_run_log_has_vehicle.txt` | §5 V-4 |
| `ex9_feature_ood.py` (pedestrian) | `results/ex9_feature_run_log.txt` | §5 V-4 |
| `ex9_feature_ood.py` (traffic light) | `results/ex9_feature_run_log_has_traffic_light.txt` | §5 V-4 |
| `ex9_feature_ood.py` (vehicle) | `results/ex9_feature_run_log_has_vehicle.txt` | §5 V-4 |
| `ex9_visualise_shift.py` | `results/ex9_visualise_run_log.txt` | §3 ODD rationale (supporting) |
| generative edit (manual, no script) | `figures/forgery_original.jpg`, `figures/forgery_doctored.png` | Additional Material, Exploratory Probe |

---
