"""Exercise 9.6 — Evaluating the MSP Baseline for OOD Detection

Uses the pedestrian model's maximum softmax probability (MSP) as an OOD score.
Treats sunny/daytime test images as in-distribution and fog/night/town as OOD.

Produces:
  - Distribution plot of OOD scores (in-dist vs each OOD scenario)
  - AUROC per OOD scenario and combined

Usage:
    python ex9_msp_baseline.py
    python ex9_msp_baseline.py --data-dir "D:/ML Safety/2026" --models-dir models
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = r"D:\ML Safety\2026"
MODELS_DIR = r"D:\ML Safety\ML_Safety\models"

TEST_DIR  = os.path.join(DATA_DIR, "test",  "test")
FOG_DIR   = os.path.join(DATA_DIR, "test-fog",     "test-fog")
NIGHT_DIR = os.path.join(DATA_DIR, "test-night",   "test-night")
TOWN_DIR  = os.path.join(DATA_DIR, "test-town-01", "test-town-01")

# Model used throughout Exercises 9.6 and 9.7
TARGET_LABEL = "has_pedestrian"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(label: str, models_dir: str) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    path = os.path.join(models_dir, f"resnet18_{label}.pth")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


def load_split_df(split_dir: str) -> pd.DataFrame:
    csv = os.path.join(split_dir, "labels.csv")
    df = pd.read_csv(csv)
    df["img_path"] = df["frame"].apply(
        lambda f: os.path.join(split_dir, "rgb-front", f"{int(f):06d}.jpg")
    )
    df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)
    return df


class SplitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["img_path"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


@torch.no_grad()
def compute_msp_scores(model: nn.Module, df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
    """Return MSP = max(p, 1-p) for each image. Higher = more in-distribution."""
    ds = SplitDataset(df, TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)
    msp_scores = []
    for batch in loader:
        batch = batch.to(DEVICE)
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        msp_scores.extend(probs.max(axis=1).tolist())
    return np.array(msp_scores)


def compute_auroc(in_dist_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC for separating in-dist (label=0) from OOD (label=1).
    OOD score = 1 - MSP so that higher score => more OOD."""
    y_true  = np.concatenate([np.zeros(len(in_dist_scores)), np.ones(len(ood_scores))])
    y_score = np.concatenate([1 - in_dist_scores, 1 - ood_scores])
    return roc_auc_score(y_true, y_score)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_score_distributions(scores_dict: dict, save_path: str = None):
    """Histogram of MSP scores for in-dist and each OOD scenario."""
    n = len(scores_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    in_dist_scores = scores_dict["In-dist (test)"]
    colors = {"In-dist (test)": "#4C72B0", "Fog": "#DD8452",
              "Night": "#55A868", "Town01": "#C44E52"}

    for ax, (name, scores) in zip(axes, scores_dict.items()):
        ax.hist(in_dist_scores, bins=50, density=True, alpha=0.6,
                color=colors.get("In-dist (test)", "blue"), label="In-dist")
        if name != "In-dist (test)":
            ax.hist(scores, bins=50, density=True, alpha=0.6,
                    color=colors.get(name, "orange"), label=name)
            ax.set_title(f"In-dist vs {name}")
        else:
            ax.set_title("In-dist (test)")
        ax.set_xlabel("MSP score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.set_xlim(0.5, 1.0)

    plt.suptitle(f"MSP Score Distributions — {TARGET_LABEL}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_roc_curves(in_dist_scores: np.ndarray, ood_scores_dict: dict,
                   auroc_dict: dict, save_path: str = None):
    """ROC curves for each OOD scenario."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for (name, ood_scores), color in zip(ood_scores_dict.items(), colors):
        y_true  = np.concatenate([np.zeros(len(in_dist_scores)), np.ones(len(ood_scores))])
        y_score = np.concatenate([1 - in_dist_scores, 1 - ood_scores])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name}  (AUROC = {auroc_dict[name]:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — MSP Baseline ({TARGET_LABEL})")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Exercise 9.6 — MSP baseline OOD detection")
    p.add_argument("--data-dir",   default=DATA_DIR)
    p.add_argument("--models-dir", default=MODELS_DIR)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-label", default=TARGET_LABEL,
                   help="Which detector's checkpoint to evaluate "
                        "(has_pedestrian, has_traffic_light, has_vehicle).")
    return p.parse_args()


def main():
    args = parse_args()

    global DATA_DIR, MODELS_DIR, TEST_DIR, FOG_DIR, NIGHT_DIR, TOWN_DIR, TARGET_LABEL
    DATA_DIR    = args.data_dir
    MODELS_DIR  = args.models_dir
    TARGET_LABEL = args.target_label
    TEST_DIR   = os.path.join(DATA_DIR, "test",  "test")
    FOG_DIR    = os.path.join(DATA_DIR, "test-fog",     "test-fog")
    NIGHT_DIR  = os.path.join(DATA_DIR, "test-night",   "test-night")
    TOWN_DIR   = os.path.join(DATA_DIR, "test-town-01", "test-town-01")

    print(f"Device     : {DEVICE}")
    print(f"Model      : {TARGET_LABEL}\n")

    # Load data
    split_dfs = {
        "In-dist (test)": load_split_df(TEST_DIR),
        "Fog":            load_split_df(FOG_DIR),
        "Night":          load_split_df(NIGHT_DIR),
        "Town01":         load_split_df(TOWN_DIR),
    }
    for name, df in split_dfs.items():
        print(f"  {name:<20}: {len(df):,} images")

    # Load model
    print(f"\nLoading {TARGET_LABEL} model ...")
    model = load_model(TARGET_LABEL, MODELS_DIR)

    # Compute MSP scores for every split
    print("Computing MSP scores ...")
    scores = {}
    for name, df in split_dfs.items():
        scores[name] = compute_msp_scores(model, df, args.batch_size)
        print(f"  {name:<20}: mean MSP = {scores[name].mean():.4f}  "
              f"std = {scores[name].std():.4f}")

    # Score distributions plot
    suffix = "" if TARGET_LABEL == "has_pedestrian" else f"_{TARGET_LABEL}"
    plot_score_distributions(
        scores,
        save_path=os.path.join(MODELS_DIR, f"ex9_msp_distributions{suffix}.png"),
    )

    # AUROC per OOD scenario
    in_dist_scores = scores["In-dist (test)"]
    ood_splits = {k: v for k, v in scores.items() if k != "In-dist (test)"}

    print("\nAUROC per OOD scenario (in-dist vs OOD):")
    auroc_dict = {}
    for name, ood_scores in ood_splits.items():
        auroc = compute_auroc(in_dist_scores, ood_scores)
        auroc_dict[name] = auroc
        print(f"  {name:<20}: AUROC = {auroc:.4f}")

    # Combined (all OOD scenarios pooled)
    all_ood_scores = np.concatenate(list(ood_splits.values()))
    combined_auroc = compute_auroc(in_dist_scores, all_ood_scores)
    print(f"\n  {'Combined (all OOD)':<20}: AUROC = {combined_auroc:.4f}")

    # ROC curves
    plot_roc_curves(
        in_dist_scores,
        ood_splits,
        auroc_dict,
        save_path=os.path.join(MODELS_DIR, f"ex9_msp_roc_curves{suffix}.png"),
    )

    # Summary
    best = max(auroc_dict, key=auroc_dict.get)
    worst = min(auroc_dict, key=auroc_dict.get)
    print(f"\nSummary:")
    print(f"  Best separation  : {best} (AUROC {auroc_dict[best]:.4f})")
    print(f"  Worst separation : {worst} (AUROC {auroc_dict[worst]:.4f})")
    print(f"  Combined AUROC   : {combined_auroc:.4f}")

    return scores, auroc_dict


if __name__ == "__main__":
    main()
