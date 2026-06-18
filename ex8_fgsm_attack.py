"""Exercise 8.4–8.5 — FGSM Adversarial Attack on CARLA Binary Classifiers

Implements FGSM for the three CARLA binary classifiers (pedestrian, traffic light,
vehicle), generates adversarial examples for ε ∈ {0.01, 0.05, 0.1}, and reports
the recall drop compared to clean inputs.

Usage:
    python ex8_fgsm_attack.py
    python ex8_fgsm_attack.py --data-dir "D:/ML Safety/2026" --models-dir models
    python ex8_fgsm_attack.py --n-samples 100
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = r"D:\ML Safety\2026"
MODELS_DIR = r"D:\ML Safety\ML_Safety\models"

TEST_DIR = os.path.join(DATA_DIR, "test", "test")

TARGET_LABELS = ["has_pedestrian", "has_traffic_light", "has_vehicle"]
EPSILONS      = [0.01, 0.05, 0.1]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN   = [0.485, 0.456, 0.406]
STD    = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Inverse transform for display (undo normalization)
INV_MEAN = [-m / s for m, s in zip(MEAN, STD)]
INV_STD  = [1.0 / s for s in STD]
INVERSE_NORMALIZE = transforms.Normalize(INV_MEAN, INV_STD)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CarlaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, label: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.label     = label
        self.transform = transform

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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(label: str, models_dir: str) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    path = os.path.join(models_dir, f"resnet18_{label}.pth")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# FGSM attack
# ---------------------------------------------------------------------------
def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    criterion: nn.Module,
) -> torch.Tensor:
    """Apply Fast Gradient Sign Method to a batch.

    Returns a perturbed tensor of the same shape — gradient graph detached.
    The perturbation is applied in the normalised pixel space (same space the
    model operates in), so ε is in units of normalised pixel values.
    """
    images = images.clone().requires_grad_(True)

    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()

    sign_grad = images.grad.data.sign()
    perturbed = images.detach() + epsilon * sign_grad
    return perturbed


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return predicted class indices for a batch (no grad)."""
    return model(images).argmax(dim=1)


def evaluate_recall(
    model: nn.Module,
    loader: DataLoader,
    epsilon: float,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Return (clean_recall, adversarial_recall) for the positive class."""
    all_labels, clean_preds, adv_preds = [], [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            clean_pred = predict(model, images)

        adv_images = fgsm_attack(model, images, labels, epsilon, criterion)
        with torch.no_grad():
            adv_pred = predict(model, adv_images)

        all_labels.extend(labels.cpu().numpy())
        clean_preds.extend(clean_pred.cpu().numpy())
        adv_preds.extend(adv_pred.cpu().numpy())

    y_true     = np.array(all_labels)
    clean_rec  = recall_score(y_true, clean_preds,  pos_label=1, zero_division=0)
    adv_rec    = recall_score(y_true, adv_preds,    pos_label=1, zero_division=0)
    return clean_rec, adv_rec


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor to a displayable HWC uint8 array."""
    img = INVERSE_NORMALIZE(t.cpu())
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def plot_adversarial_examples(
    model: nn.Module,
    loader: DataLoader,
    label: str,
    epsilons: list[float],
    save_path: str = None,
):
    """Show a clean image and its adversarial counterpart for each epsilon."""
    criterion = nn.CrossEntropyLoss()

    # Grab one batch
    images, labels = next(iter(loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    # Pick one positive example for visual clarity
    pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
    idx = pos_indices[0].item() if len(pos_indices) > 0 else 0

    clean_img = images[idx:idx+1]
    clean_lbl = labels[idx:idx+1]

    n_cols = 1 + len(epsilons)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # Clean image
    with torch.no_grad():
        clean_pred = model(clean_img).argmax(dim=1).item()
    axes[0].imshow(tensor_to_image(clean_img[0]))
    axes[0].set_title(f"Clean\npred={clean_pred}, true={clean_lbl.item()}")
    axes[0].axis("off")

    # Adversarial images
    for ax, eps in zip(axes[1:], epsilons):
        adv_img = fgsm_attack(model, clean_img, clean_lbl, eps, criterion)
        with torch.no_grad():
            adv_pred = model(adv_img).argmax(dim=1).item()

        # Amplify perturbation for display
        diff = (adv_img - clean_img).abs().cpu()
        diff_display = (diff * 10).clamp(0, 1)  # amplify × 10 for visibility

        ax.imshow(tensor_to_image(adv_img[0]))
        ax.set_title(f"ε = {eps}\npred={adv_pred}, true={clean_lbl.item()}")
        ax.axis("off")

    fig.suptitle(f"FGSM Examples — {label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_recall_table(results: dict, save_path: str = None):
    """Bar chart of recall drop per model per epsilon."""
    n_labels = len(results)
    n_eps    = len(EPSILONS)

    fig, axes = plt.subplots(1, n_labels, figsize=(6 * n_labels, 5), sharey=True)
    if n_labels == 1:
        axes = [axes]

    for ax, (label, eps_data) in zip(axes, results.items()):
        clean_rec = eps_data["clean"]
        adv_recs  = [eps_data[eps] for eps in EPSILONS]

        x = np.arange(n_eps)
        width = 0.35
        ax.bar(x - width / 2, [clean_rec] * n_eps, width, label="Clean", color="#4C72B0")
        ax.bar(x + width / 2, adv_recs,             width, label="Adversarial", color="#C44E52")

        ax.set_xticks(x)
        ax.set_xticklabels([f"ε={e}" for e in EPSILONS])
        ax.set_ylabel("Recall (positive class)")
        ax.set_ylim(0, 1.1)
        ax.set_title(label.replace("has_", "").capitalize())
        ax.legend()

    fig.suptitle("Recall: Clean vs FGSM Adversarial", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Exercise 8.4–8.5 — FGSM attack on CARLA models")
    p.add_argument("--data-dir",   default=DATA_DIR)
    p.add_argument("--models-dir", default=MODELS_DIR)
    p.add_argument("--n-samples",  type=int, default=None,
                   help="Use a random subset of N images (None = full test set)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    global DATA_DIR, MODELS_DIR, TEST_DIR
    DATA_DIR   = args.data_dir
    MODELS_DIR = args.models_dir
    TEST_DIR   = os.path.join(DATA_DIR, "test", "test")
    img_dir    = os.path.join(TEST_DIR, "rgb-front")

    print(f"Device     : {DEVICE}")
    print(f"Test split : {TEST_DIR}")
    print(f"Models dir : {MODELS_DIR}\n")

    os.makedirs(MODELS_DIR, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    # Load full test CSV once
    test_df = pd.read_csv(os.path.join(TEST_DIR, "labels.csv"))
    test_df["_img_path"] = test_df["frame"].apply(
        lambda f: os.path.join(img_dir, f"{int(f):06d}.jpg")
    )
    test_df = test_df[test_df["_img_path"].apply(os.path.exists)].reset_index(drop=True)

    if args.n_samples and args.n_samples < len(test_df):
        test_df = test_df.sample(args.n_samples, random_state=args.seed).reset_index(drop=True)

    print(f"Using {len(test_df):,} test images\n")

    results = {}

    for label in TARGET_LABELS:
        print(f"{'='*60}")
        print(f"Model: {label}")
        print(f"{'='*60}")

        model = load_model(label, MODELS_DIR)

        dataset = CarlaDataset(test_df, img_dir, label, TRANSFORM)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # ---- Exercise 8.4: Visualise adversarial examples ----
        print("  Generating adversarial example visualisations ...")
        plot_adversarial_examples(
            model, loader, label, EPSILONS,
            save_path=os.path.join(MODELS_DIR, f"ex8_fgsm_{label}.png"),
        )

        # ---- Exercise 8.5: Recall drop ----
        label_results = {}
        print(f"\n  Recall on positive class (label = 1):")

        for eps in EPSILONS:
            clean_rec, adv_rec = evaluate_recall(model, loader, eps, criterion)
            label_results[eps] = adv_rec
            if "clean" not in label_results:
                label_results["clean"] = clean_rec
            drop = clean_rec - adv_rec
            print(f"    ε = {eps:4.2f}  |  Clean recall = {clean_rec:.4f}  "
                  f"Adv recall = {adv_rec:.4f}  |  Drop = {drop:+.4f}")

        results[label] = label_results
        print()

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print("RECALL DROP SUMMARY")
    print(f"{'='*60}")
    header = f"{'Label':<25} {'Clean':>8}" + "".join(f"  ε={e:4.2f}" for e in EPSILONS)
    print(header)
    print("-" * len(header))
    for label, eps_data in results.items():
        clean = eps_data["clean"]
        row   = f"{label:<25} {clean:>8.4f}"
        for eps in EPSILONS:
            drop = clean - eps_data[eps]
            row += f"  {drop:+6.4f}"
        print(row)

    # ---- Recall drop bar chart ----
    plot_recall_table(
        results,
        save_path=os.path.join(MODELS_DIR, "ex8_recall_drop.png"),
    )

    return results


if __name__ == "__main__":
    main()
