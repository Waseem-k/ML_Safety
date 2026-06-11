"""Exercise 9.4 — Visualising the Distribution Shift

Displays sample images from each data split side-by-side and computes the mean
softmax confidence of all three ResNet-18 models on in-distribution vs OOD inputs.

Usage:
    python ex9_visualise_shift.py
    python ex9_visualise_shift.py --data-dir "D:/ML Safety/2026" --models-dir models
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

# ---------------------------------------------------------------------------
# Configuration — adjust paths if your data lives elsewhere
# ---------------------------------------------------------------------------
DATA_DIR   = r"D:\ML Safety\2026"
MODELS_DIR = r"D:\ML Safety\ML_Safety\models"

# In-distribution splits (nested: <split>/<split>/)
TRAIN_DIR = os.path.join(DATA_DIR, "train", "train")
TEST_DIR  = os.path.join(DATA_DIR, "test",  "test")

# OOD splits (flat: <split>/)
FOG_DIR   = os.path.join(DATA_DIR, "test-fog",    "test-fog")
NIGHT_DIR = os.path.join(DATA_DIR, "test-night",   "test-night")
TOWN_DIR  = os.path.join(DATA_DIR, "test-town-01", "test-town-01")

LABELS = ["has_pedestrian", "has_traffic_light", "has_vehicle"]
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


def load_image_raw(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def denorm(tensor):
    img = tensor.permute(1, 2, 0).numpy()
    return np.clip(img * np.array(STD) + np.array(MEAN), 0, 1)


class SplitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["img_path"]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


@torch.no_grad()
def mean_confidence(model: nn.Module, df: pd.DataFrame, batch_size: int = 64) -> float:
    """Mean max-softmax-probability over a split (higher = more confident)."""
    ds = SplitDataset(df, TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)
    probs_all = []
    for batch in loader:
        batch = batch.to(DEVICE)
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        msp = probs.max(axis=1)
        probs_all.extend(msp.tolist())
    return float(np.mean(probs_all))


# ---------------------------------------------------------------------------
# Exercise 9.4.1 — Display sample images from each split
# ---------------------------------------------------------------------------
def plot_sample_images(split_dfs: dict, n: int = 5, save_path: str = None):
    split_names = list(split_dfs.keys())
    fig, axes = plt.subplots(len(split_names), n, figsize=(3 * n, 3 * len(split_names)))

    for row, name in enumerate(split_names):
        df = split_dfs[name]
        sample = df.sample(min(n, len(df)), random_state=42)
        for col, (_, row_data) in enumerate(sample.iterrows()):
            ax = axes[row][col]
            ax.imshow(load_image_raw(row_data["img_path"]))
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold", rotation=0,
                              labelpad=80, va="center")
        # pad remaining columns if sample < n
        for col in range(len(sample), n):
            axes[row][col].axis("off")

    plt.suptitle("Sample Images by Split", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Exercise 9.4.3 — Compute mean softmax confidence per model per split
# ---------------------------------------------------------------------------
def compute_confidence_table(models_dict: dict, split_dfs: dict) -> pd.DataFrame:
    rows = []
    for split_name, df in split_dfs.items():
        row = {"Split": split_name}
        for label, model in models_dict.items():
            conf = mean_confidence(model, df)
            row[label] = round(conf, 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Split")


def plot_confidence_table(df_conf: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(df_conf))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, label in enumerate(df_conf.columns):
        ax.bar(x + i * width, df_conf[label], width, label=label, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df_conf.index, rotation=15, ha="right")
    ax.set_ylabel("Mean MSP (max softmax probability)")
    ax.set_title("Mean Softmax Confidence per Model per Split")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.6, label="Chance (0.5)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Exercise 9.4 — Distribution shift visualisation")
    p.add_argument("--data-dir",   default=DATA_DIR,   help="Root dataset directory")
    p.add_argument("--models-dir", default=MODELS_DIR, help="Directory containing .pth files")
    p.add_argument("--n-images",   type=int, default=5, help="Images per split to display")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    global DATA_DIR, MODELS_DIR, TRAIN_DIR, TEST_DIR, FOG_DIR, NIGHT_DIR, TOWN_DIR
    DATA_DIR   = args.data_dir
    MODELS_DIR = args.models_dir
    TRAIN_DIR  = os.path.join(DATA_DIR, "train", "train")
    TEST_DIR   = os.path.join(DATA_DIR, "test",  "test")
    FOG_DIR    = os.path.join(DATA_DIR, "test-fog",     "test-fog")
    NIGHT_DIR  = os.path.join(DATA_DIR, "test-night",   "test-night")
    TOWN_DIR   = os.path.join(DATA_DIR, "test-town-01", "test-town-01")

    print(f"Device    : {DEVICE}")
    print(f"Data dir  : {DATA_DIR}")
    print(f"Models dir: {MODELS_DIR}\n")

    # Load split dataframes
    split_dfs = {
        "In-dist (test)": load_split_df(TEST_DIR),
        "Fog":            load_split_df(FOG_DIR),
        "Night":          load_split_df(NIGHT_DIR),
        "Town01":         load_split_df(TOWN_DIR),
    }
    for name, df in split_dfs.items():
        print(f"  {name:<20}: {len(df):,} images")

    # ── Exercise 9.4.1 & 9.4.2 — sample image grid ──
    print("\nPlotting sample images ...")
    plot_sample_images(
        split_dfs,
        n=args.n_images,
        save_path=os.path.join(MODELS_DIR, "ex9_sample_images.png"),
    )

    # ── Exercise 9.4.3 — confidence per model per split ──
    print("\nLoading models ...")
    model_dict = {label: load_model(label, MODELS_DIR) for label in LABELS}

    print("Computing mean softmax confidence (this may take a minute) ...")
    conf_df = compute_confidence_table(model_dict, split_dfs)

    print("\nMean MSP per model per split:")
    print(conf_df.to_string())

    plot_confidence_table(
        conf_df,
        save_path=os.path.join(MODELS_DIR, "ex9_confidence_per_split.png"),
    )

    # Interpret direction
    print("\nInterpretation:")
    in_dist_mean = conf_df.loc["In-dist (test)"].mean()
    for split in ["Fog", "Night", "Town01"]:
        if split in conf_df.index:
            ood_mean = conf_df.loc[split].mean()
            direction = "LOWER" if ood_mean < in_dist_mean else "HIGHER"
            print(f"  {split:<10}: mean conf = {ood_mean:.4f} ({direction} than in-dist {in_dist_mean:.4f})")


if __name__ == "__main__":
    main()
