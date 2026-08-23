"""Exercise 4 — Model Testing and Validation

Covers the practical parts of Exercise Sheet 4 (030_testing.pdf):
  4.5  ODD coverage via k-projection coverage (k = 1, 2, 3)
  4.7  Per-class evaluation: precision / recall / F1 + confusion matrix
       for all three CARLA binary classifiers on the clean in-distribution
       test split.

This is the evidence source for:
  - Report Section 3.2 (ODD Coverage)               <- k-projection table
  - Report Section 5, V-1 (In-Distribution Accuracy)  <- per-class metrics table

Usage:
    python ex4_testing.py
    python ex4_testing.py --data-dir "D:/ML Safety/2026" --models-dir models
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from odd_coverage.kprojection import KProjectionCoverage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = r"D:\ML Safety\2026"
MODELS_DIR = r"D:\ML Safety\ML_Safety\models"
RESULTS_DIR = r"D:\ML Safety\ML_Safety\results"

TEST_DIR = os.path.join(DATA_DIR, "test", "test")
IMG_DIR  = os.path.join(TEST_DIR, "rgb-front")

TARGET_LABELS = ["has_pedestrian", "has_traffic_light", "has_vehicle"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Suggested safety thresholds (report Section 5, V-1) — see Exercise 4.7.4 /
# Exercise 8 recall-drop discussion for justification.
RECALL_THRESHOLDS = {
    "has_pedestrian":    0.90,
    "has_traffic_light": 0.85,
    "has_vehicle":       0.85,
}


# ---------------------------------------------------------------------------
# Dataset / model loading (same conventions as ex7/ex8/ex9)
# ---------------------------------------------------------------------------
class CarlaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, label: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir    = img_dir
        self.label      = label
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{int(row['frame']):06d}.jpg")
        image    = Image.open(img_path).convert("RGB")
        target   = int(bool(row[self.label]))
        if self.transform:
            image = self.transform(image)
        return image, target


def load_model(label: str, models_dir: str) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    path = os.path.join(models_dir, f"resnet18_{label}.pth")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# 4.7 — Per-class evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(label: str, test_df: pd.DataFrame, models_dir: str, batch_size: int = 32):
    model   = load_model(label, models_dir)
    dataset = CarlaDataset(test_df, IMG_DIR, label, TRANSFORM)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true, y_pred = [], []
    for images, targets in loader:
        images = images.to(DEVICE)
        preds  = model(images).argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(targets.numpy().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    threshold = RECALL_THRESHOLDS[label]
    positive_recall = recall[1]

    return {
        "label": label,
        "n_test": len(y_true),
        "n_positive": int(y_true.sum()),
        "precision_neg": float(precision[0]), "precision_pos": float(precision[1]),
        "recall_neg":    float(recall[0]),    "recall_pos":    float(recall[1]),
        "f1_neg":        float(f1[0]),        "f1_pos":        float(f1[1]),
        "confusion_matrix": cm.tolist(),  # rows = true [0,1], cols = pred [0,1]
        "threshold": threshold,
        "met": bool(positive_recall >= threshold),
    }


def plot_confusion_matrix(result: dict, save_path: str):
    cm = np.array(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"], ax=ax)
    ax.set_title(result["label"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4.5 — k-projection ODD coverage
# ---------------------------------------------------------------------------
def bin_steer(s: float) -> str:
    if s < -0.05:
        return "left"
    if s > 0.05:
        return "right"
    return "straight"


def bin_dynamics(row) -> str:
    if row["brake"] > 0.05:
        return "braking"
    if row["throttle"] > 0.05:
        return "accelerating"
    return "coasting"


def bin_size(px: float) -> str:
    # object-size proxy (near/far/absent), derived from px_* columns in labels.csv
    if px <= 0:
        return "absent"
    if px < 500:
        return "small"
    return "large"


def build_scenarios(test_df: pd.DataFrame, actions_df: pd.DataFrame) -> list[dict]:
    """Build one ODD scenario per test frame.

    Weather and lighting are intentionally excluded: within the in-distribution
    test split they are constant (single simulated drive, see Section 3.1 ODD
    Gap Analysis) — including them would 100%-cover trivially and inflate the
    reported coverage. Instead we use the scene-content and ego-dynamics
    dimensions that actually vary frame-to-frame, which is what the perception
    + planning stack must actually generalise over within its nominal ODD.
    """
    # NOTE: labels.csv 'frame' is the original CARLA simulation tick (sampled
    # every 10th tick, e.g. 0, 10, 20, ...), while actions.feather 'frame' is
    # just the row index (0, 1, 2, ...) of the same sampled sequence. The two
    # files are positionally aligned, not joinable on the 'frame' column.
    assert len(test_df) == len(actions_df), "labels/actions row count mismatch"
    merged = pd.concat(
        [test_df.reset_index(drop=True), actions_df.reset_index(drop=True).drop(columns=["frame"])],
        axis=1,
    )
    scenarios = []
    for _, row in merged.iterrows():
        scenarios.append({
            "pedestrian_present":    bool(row["has_pedestrian"]),
            "traffic_light_present": bool(row["has_traffic_light"]),
            "vehicle_present":       bool(row["has_vehicle"]),
            "pedestrian_size":       bin_size(row["px_pedestrian"]),
            "vehicle_size":          bin_size(row["px_vehicle"]),
            "steer_regime":          bin_steer(row["steer"]),
            "dynamics_regime":       bin_dynamics(row),
        })
    return scenarios


def compute_k_projection_coverage(scenarios: list[dict]) -> dict:
    desc = {
        "pedestrian_present":    [False, True],
        "traffic_light_present": [False, True],
        "vehicle_present":       [False, True],
        "pedestrian_size":       ["absent", "small", "large"],
        "vehicle_size":          ["absent", "small", "large"],
        "steer_regime":          ["left", "straight", "right"],
        "dynamics_regime":       ["braking", "accelerating", "coasting"],
    }

    results = {}
    for k in (1, 2, 3):
        cov = KProjectionCoverage(k=k, desc=desc)
        cov.add_scenarios(scenarios)
        r = cov.compute()
        results[k] = {
            "coverage": r.coverage, "covered": r.covered,
            "total": r.total, "scenes": r.scenes,
        }
    return {"dimensions": desc, "results": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Exercise 4 — testing & ODD coverage")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    global DATA_DIR, MODELS_DIR, TEST_DIR, IMG_DIR
    DATA_DIR   = args.data_dir
    MODELS_DIR = args.models_dir
    TEST_DIR   = os.path.join(DATA_DIR, "test", "test")
    IMG_DIR    = os.path.join(TEST_DIR, "rgb-front")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Device     : {DEVICE}")
    print(f"Test split : {TEST_DIR}")
    print(f"Models dir : {MODELS_DIR}\n")

    test_df = pd.read_csv(os.path.join(TEST_DIR, "labels.csv"))

    # ---- 4.7: per-class evaluation ----
    print(f"{'='*70}\nExercise 4.7 — Per-class evaluation\n{'='*70}")
    all_results = {}
    for label in TARGET_LABELS:
        print(f"\nEvaluating {label} ...")
        result = evaluate_model(label, test_df, MODELS_DIR)
        all_results[label] = result

        plot_confusion_matrix(result, os.path.join(MODELS_DIR, f"ex4_confusion_{label}.png"))

        print(f"  n_test={result['n_test']}  n_positive={result['n_positive']}")
        print(f"  Precision (pos): {result['precision_pos']:.4f}")
        print(f"  Recall    (pos): {result['recall_pos']:.4f}   "
              f"(threshold={result['threshold']:.2f}, met={result['met']})")
        print(f"  F1        (pos): {result['f1_pos']:.4f}")
        print(f"  Confusion matrix [[TN,FP],[FN,TP]]: {result['confusion_matrix']}")

    print(f"\n{'='*70}\nSUMMARY (positive-class recall = safety-relevant metric)\n{'='*70}")
    print(f"{'Model':<20}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Threshold':>12}{'Met?':>8}")
    for label, r in all_results.items():
        print(f"{label:<20}{r['precision_pos']:>12.4f}{r['recall_pos']:>12.4f}"
              f"{r['f1_pos']:>12.4f}{r['threshold']:>12.2f}{str(r['met']):>8}")

    with open(os.path.join(RESULTS_DIR, "ex4_per_class_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- 4.5: k-projection ODD coverage ----
    print(f"\n{'='*70}\nExercise 4.5 — k-projection ODD coverage\n{'='*70}")
    actions_df = pd.read_feather(os.path.join(TEST_DIR, "actions.feather"))
    scenarios = build_scenarios(test_df, actions_df)
    coverage = compute_k_projection_coverage(scenarios)

    print(f"Scenarios: {len(scenarios)}")
    print(f"{'k':<5}{'Coverage':>12}{'Covered':>12}{'Total':>12}")
    for k, r in coverage["results"].items():
        print(f"{k:<5}{r['coverage']:>12.4f}{r['covered']:>12}{r['total']:>12}")

    with open(os.path.join(RESULTS_DIR, "ex4_kprojection_coverage.json"), "w") as f:
        json.dump(coverage, f, indent=2)

    print(f"\nSaved: {RESULTS_DIR}\\ex4_per_class_metrics.json")
    print(f"Saved: {RESULTS_DIR}\\ex4_kprojection_coverage.json")
    print(f"Saved: {MODELS_DIR}\\ex4_confusion_<label>.png (x3)")


if __name__ == "__main__":
    main()
