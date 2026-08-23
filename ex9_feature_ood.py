"""Exercise 9.7 — Feature-Based OOD Detection (Mahalanobis Distance)

Implements a Mahalanobis distance OOD detector using deep features from the
avgpool layer of the pedestrian ResNet-18 (the same model as Exercise 9.6).

Steps:
  1. Extract 512-dim features from training images (in-distribution, fit set)
  2. Fit class-conditional Gaussian with shared (tied) covariance on training features
  3. Score all test splits — larger distance => more OOD
  4. Compute AUROC and compare to MSP baseline from Exercise 9.6

Usage:
    python ex9_feature_ood.py
    python ex9_feature_ood.py --data-dir "D:/ML Safety/2026" --models-dir models
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = r"D:\ML Safety\2026"
MODELS_DIR = r"D:\ML Safety\ML_Safety\models"

TRAIN_DIR = os.path.join(DATA_DIR, "train", "train")
TEST_DIR  = os.path.join(DATA_DIR, "test",  "test")
FOG_DIR   = os.path.join(DATA_DIR, "test-fog",     "test-fog")
NIGHT_DIR = os.path.join(DATA_DIR, "test-night",   "test-night")
TOWN_DIR  = os.path.join(DATA_DIR, "test-town-01", "test-town-01")

TARGET_LABEL = "has_pedestrian"
FEATURE_DIM  = 512   # ResNet-18 avgpool output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_split_df(split_dir: str) -> pd.DataFrame:
    csv = os.path.join(split_dir, "labels.csv")
    df = pd.read_csv(csv)
    df["img_path"] = df["frame"].apply(
        lambda f: os.path.join(split_dir, "rgb-front", f"{int(f):06d}.jpg")
    )
    df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)
    return df


class SplitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str = None, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.label_col is not None:
            label = int(bool(row[self.label_col]))
            return img, label
        return img


# ---------------------------------------------------------------------------
# Model and feature extraction
# ---------------------------------------------------------------------------
def load_model(label: str, models_dir: str) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    path = os.path.join(models_dir, f"resnet18_{label}.pth")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


@torch.no_grad()
def extract_features(model: nn.Module, df: pd.DataFrame,
                     label_col: str = None, batch_size: int = 64):
    """Extract avgpool features (512-dim) and optionally labels."""
    ds = SplitDataset(df, label_col=label_col, transform=TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)

    feature_list = []
    label_list = []

    # Hook the avgpool layer to capture 512-dim features before the FC head
    captured = {}

    def _hook(module, input, output):
        # output: (B, 512, 1, 1) → flatten to (B, 512)
        captured["features"] = output.squeeze(-1).squeeze(-1).detach().cpu().numpy()

    hook = model.avgpool.register_forward_hook(_hook)

    for batch in loader:
        if label_col is not None:
            imgs, labels = batch
            label_list.extend(labels.numpy().tolist())
        else:
            imgs = batch

        imgs = imgs.to(DEVICE)
        _ = model(imgs)  # forward pass triggers the hook
        feature_list.append(captured["features"].copy())

    hook.remove()

    features = np.vstack(feature_list)
    labels = np.array(label_list) if label_col is not None else None
    return features, labels


# ---------------------------------------------------------------------------
# MSP scoring (replicated from ex9_msp_baseline.py for standalone use)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_msp_scores(model: nn.Module, df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
    ds = SplitDataset(df, transform=TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)
    msp = []
    for batch in loader:
        batch = batch.to(DEVICE)
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        msp.extend(probs.max(axis=1).tolist())
    return np.array(msp)


def compute_auroc_ood(in_dist_scores, ood_scores, higher_is_ood: bool = True):
    """AUROC: in-dist = label 0, OOD = label 1."""
    y_true  = np.concatenate([np.zeros(len(in_dist_scores)), np.ones(len(ood_scores))])
    if higher_is_ood:
        y_score = np.concatenate([in_dist_scores, ood_scores])
    else:
        # MSP: higher MSP = more in-dist, so flip
        y_score = np.concatenate([-in_dist_scores, -ood_scores])
    return roc_auc_score(y_true, y_score)


# ---------------------------------------------------------------------------
# Mahalanobis detector
# ---------------------------------------------------------------------------
class MahalanobisDetector:
    """
    Class-conditional Gaussian OOD detector with shared covariance.
    Larger score => more OOD.
    """

    def __init__(self):
        self.class_means = {}
        self.cov_estimator = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        classes = np.unique(labels)
        centered_parts = []

        for c in classes:
            mask = labels == c
            class_feats = features[mask]
            mu = class_feats.mean(axis=0)
            self.class_means[c] = mu
            centered_parts.append(class_feats - mu)

        # Pool centred features from all classes and fit shared covariance
        X_centered = np.vstack(centered_parts)
        self.cov_estimator = EmpiricalCovariance(assume_centered=True)
        self.cov_estimator.fit(X_centered)
        print(f"  Fitted Mahalanobis detector on {len(features):,} samples, "
              f"{len(classes)} classes.")

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return min Mahalanobis distance to any class centroid (squared)."""
        distances = np.stack([
            self.cov_estimator.mahalanobis(features - mu)
            for mu in self.class_means.values()
        ], axis=1)   # shape: (N, num_classes)
        return distances.min(axis=1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_score_comparison(in_dist_mah, ood_dict_mah,
                          in_dist_msp, ood_dict_msp,
                          save_path: str = None):
    scenarios = list(ood_dict_mah.keys())
    fig, axes = plt.subplots(2, len(scenarios), figsize=(5 * len(scenarios), 8))

    colors = {"Fog": "#DD8452", "Night": "#55A868", "Town01": "#C44E52"}

    for col, name in enumerate(scenarios):
        # Top row: Mahalanobis
        ax = axes[0][col]
        ax.hist(in_dist_mah, bins=60, density=True, alpha=0.6,
                color="#4C72B0", label="In-dist")
        ax.hist(ood_dict_mah[name], bins=60, density=True, alpha=0.6,
                color=colors.get(name, "gray"), label=name)
        ax.set_title(f"Mahalanobis — {name}")
        ax.set_xlabel("Distance score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        # Bottom row: MSP (inverted so higher = more OOD for visual consistency)
        ax = axes[1][col]
        ax.hist(1 - in_dist_msp, bins=50, density=True, alpha=0.6,
                color="#4C72B0", label="In-dist")
        ax.hist(1 - ood_dict_msp[name], bins=50, density=True, alpha=0.6,
                color=colors.get(name, "gray"), label=name)
        ax.set_title(f"MSP (1−score) — {name}")
        ax.set_xlabel("OOD score (1 − MSP)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle(f"Mahalanobis vs MSP Score Distributions — {TARGET_LABEL}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_auroc_comparison(auroc_msp: dict, auroc_mah: dict, save_path: str = None):
    scenarios = list(auroc_msp.keys())
    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_msp = ax.bar(x - width / 2, [auroc_msp[s] for s in scenarios],
                      width, label="MSP baseline", color="#4C72B0", alpha=0.85)
    bars_mah = ax.bar(x + width / 2, [auroc_mah[s] for s in scenarios],
                      width, label="Mahalanobis", color="#DD8452", alpha=0.85)

    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.6, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"AUROC Comparison: MSP vs Mahalanobis — {TARGET_LABEL}")
    ax.legend()

    # Annotate bars
    for bar in list(bars_msp) + list(bars_mah):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_roc_curves_comparison(in_dist_mah, in_dist_msp,
                               ood_dict_mah, ood_dict_msp,
                               save_path: str = None):
    scenarios = list(ood_dict_mah.keys())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, name in zip(axes, scenarios):
        # Mahalanobis ROC
        y_true  = np.concatenate([np.zeros(len(in_dist_mah)), np.ones(len(ood_dict_mah[name]))])
        y_score = np.concatenate([in_dist_mah, ood_dict_mah[name]])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc_m = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, color="#DD8452", label=f"Mahalanobis ({auroc_m:.3f})")

        # MSP ROC
        y_score_msp = np.concatenate([1 - in_dist_msp, 1 - ood_dict_msp[name]])
        fpr_m, tpr_m, _ = roc_curve(y_true[:len(in_dist_msp) + len(ood_dict_msp[name])],
                                     y_score_msp)
        auroc_msp = roc_auc_score(
            np.concatenate([np.zeros(len(in_dist_msp)), np.ones(len(ood_dict_msp[name]))]),
            y_score_msp,
        )
        ax.plot(fpr_m, tpr_m, lw=2, color="#4C72B0", ls="--",
                label=f"MSP ({auroc_msp:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_title(f"ROC — {name}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle(f"ROC Curves Comparison — {TARGET_LABEL}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Exercise 9.7 — Mahalanobis OOD detection")
    p.add_argument("--data-dir",    default=DATA_DIR)
    p.add_argument("--models-dir",  default=MODELS_DIR)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--train-limit", type=int, default=None,
                   help="Max training samples to use for fitting (None = all). "
                        "Use a smaller number to speed up on large datasets.")
    p.add_argument("--target-label", default=TARGET_LABEL,
                   help="Which detector's checkpoint/features to evaluate "
                        "(has_pedestrian, has_traffic_light, has_vehicle).")
    return p.parse_args()


def main():
    args = parse_args()

    global DATA_DIR, MODELS_DIR, TRAIN_DIR, TEST_DIR, FOG_DIR, NIGHT_DIR, TOWN_DIR, TARGET_LABEL
    DATA_DIR    = args.data_dir
    MODELS_DIR  = args.models_dir
    TARGET_LABEL = args.target_label
    TRAIN_DIR  = os.path.join(DATA_DIR, "train", "train")
    TEST_DIR   = os.path.join(DATA_DIR, "test",  "test")
    FOG_DIR    = os.path.join(DATA_DIR, "test-fog",     "test-fog")
    NIGHT_DIR  = os.path.join(DATA_DIR, "test-night",   "test-night")
    TOWN_DIR   = os.path.join(DATA_DIR, "test-town-01", "test-town-01")

    print(f"Device     : {DEVICE}")
    print(f"Model      : {TARGET_LABEL}\n")

    model = load_model(TARGET_LABEL, MODELS_DIR)

    # ── 1. Extract training features (fit set) ──────────────────────────────
    train_df = load_split_df(TRAIN_DIR)
    if args.train_limit:
        train_df = train_df.sample(min(args.train_limit, len(train_df)), random_state=42)
    print(f"Extracting training features ({len(train_df):,} images) ...")
    train_feats, train_labels = extract_features(
        model, train_df, label_col=TARGET_LABEL, batch_size=args.batch_size
    )
    print(f"  Feature shape: {train_feats.shape}")
    print(f"  Class distribution: {np.bincount(train_labels)}")

    # ── 2. Fit Mahalanobis detector ─────────────────────────────────────────
    print("\nFitting Mahalanobis detector ...")
    detector = MahalanobisDetector()
    detector.fit(train_feats, train_labels)

    # ── 3. Extract test features and compute scores ──────────────────────────
    split_dirs = {
        "In-dist (test)": TEST_DIR,
        "Fog":            FOG_DIR,
        "Night":          NIGHT_DIR,
        "Town01":         TOWN_DIR,
    }

    print("\nScoring test splits ...")
    mah_scores = {}
    msp_scores = {}
    for name, path in split_dirs.items():
        df = load_split_df(path)
        feats, _ = extract_features(model, df, batch_size=args.batch_size)
        mah_scores[name] = detector.score(feats)
        msp_scores[name] = compute_msp_scores(model, df, args.batch_size)
        print(f"  {name:<20}: Mah mean = {mah_scores[name].mean():.2f}  "
              f"MSP mean = {msp_scores[name].mean():.4f}")

    # ── 4. Compute AUROC ─────────────────────────────────────────────────────
    in_dist_mah = mah_scores["In-dist (test)"]
    in_dist_msp = msp_scores["In-dist (test)"]
    ood_splits = [("Fog", "Fog"), ("Night", "Night"), ("Town01", "Town01")]

    auroc_mah = {}
    auroc_msp = {}

    print("\nAUROC comparison:")
    print(f"  {'Scenario':<20}  {'MSP AUROC':>10}  {'Mah AUROC':>10}  {'Gap':>8}")
    print("  " + "-" * 55)

    for display, key in ood_splits:
        if key not in mah_scores:
            continue
        a_mah = compute_auroc_ood(in_dist_mah, mah_scores[key], higher_is_ood=True)
        a_msp = compute_auroc_ood(in_dist_msp, msp_scores[key], higher_is_ood=False)
        auroc_mah[display] = a_mah
        auroc_msp[display] = a_msp
        gap = a_mah - a_msp
        print(f"  {display:<20}  {a_msp:>10.4f}  {a_mah:>10.4f}  {gap:>+8.4f}")

    # Which scenario has the largest gap?
    if auroc_mah:
        gaps = {k: auroc_mah[k] - auroc_msp[k] for k in auroc_mah}
        best_gap_scenario = max(gaps, key=gaps.get)
        print(f"\n  Largest Mahalanobis improvement: {best_gap_scenario} "
              f"(gap = {gaps[best_gap_scenario]:+.4f})")

    # ── 5. Plots ─────────────────────────────────────────────────────────────
    suffix = "" if TARGET_LABEL == "has_pedestrian" else f"_{TARGET_LABEL}"
    ood_mah = {k: mah_scores[k] for k in auroc_mah}
    ood_msp = {k: msp_scores[k] for k in auroc_msp}

    plot_score_comparison(
        in_dist_mah, ood_mah,
        in_dist_msp, ood_msp,
        save_path=os.path.join(MODELS_DIR, f"ex9_feature_distributions{suffix}.png"),
    )
    plot_auroc_comparison(
        auroc_msp, auroc_mah,
        save_path=os.path.join(MODELS_DIR, f"ex9_auroc_comparison{suffix}.png"),
    )
    plot_roc_curves_comparison(
        in_dist_mah, in_dist_msp,
        ood_mah, ood_msp,
        save_path=os.path.join(MODELS_DIR, f"ex9_roc_comparison{suffix}.png"),
    )


if __name__ == "__main__":
    main()
