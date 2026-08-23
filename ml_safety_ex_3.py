import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Paths — change DATA_DIR to wherever your dataset lives
# ---------------------------------------------------------------------------
DATA_DIR = r"D:\ML Safety\2026"

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "train", "rgb-front")
TRAIN_CSV = os.path.join(DATA_DIR, "train", "train", "labels.csv")
TEST_CSV = os.path.join(DATA_DIR, "test", "test", "labels.csv")

LABELS = ["has_pedestrian", "has_traffic_light", "has_vehicle"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CarlaDataset(Dataset):
    def __init__(self, df, img_dir, target_label, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.target_label = target_label
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{str(row['frame']).zfill(6)}.jpg")
        image = Image.open(img_path).convert("RGB")
        label = int(bool(row[self.target_label]))
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def show_class_distribution(train_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, label in enumerate(LABELS):
        sns.countplot(data=train_df, x=label, ax=axes[i])
        axes[i].set_title(f"Distribution of {label}")
    plt.tight_layout()
    plt.show()


def show_examples(df, img_dir, label, n=3):
    positives = df[df[label].astype(bool)]
    if len(positives) < n:
        print(f"Not enough positive samples for {label} (found {len(positives)})")
        return
    sample = positives.sample(n)
    fig, axes = plt.subplots(1, n, figsize=(12, 4))
    for ax, (_, row) in zip(axes, sample.iterrows()):
        img_path = os.path.join(img_dir, f"{str(row['frame']).zfill(6)}.jpg")
        ax.imshow(Image.open(img_path))
        ax.axis("off")
        ax.set_title(f"{label} Present")
    plt.suptitle(label)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_binary_model(train_df, target_label, epochs=5, save_dir=DATA_DIR):
    print(f"\n--- Training Model for: {target_label} ---")

    dataset = CarlaDataset(train_df, TRAIN_IMG_DIR, target_label, TRANSFORM)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    loss_history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history, marker="o", label="Training Loss")
    plt.title(f"Loss Curve: {target_label}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    save_path = os.path.join(save_dir, f"resnet18_{target_label}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="ML Safety Exercise 3 — CARLA multi-label classifier")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Root directory of the dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per model")
    parser.add_argument("--skip-eda", action="store_true", help="Skip the EDA visualisations")
    return parser.parse_args()


def main():
    args = parse_args()

    global DATA_DIR, TRAIN_IMG_DIR, TRAIN_CSV, TEST_CSV
    DATA_DIR = args.data_dir
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "train", "rgb-front")
    TRAIN_CSV = os.path.join(DATA_DIR, "train", "train", "labels.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test", "test", "labels.csv")

    print(f"Using device: {DEVICE}")
    print(f"Dataset root: {DATA_DIR}")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"Training images: {len(train_df)}")
    print(f"Test images:     {len(test_df)}")

    if not args.skip_eda:
        show_class_distribution(train_df)
        print("Example images:")
        for label in LABELS:
            show_examples(train_df, TRAIN_IMG_DIR, label)

    trained_models = {}
    for label in LABELS:
        trained_models[label] = train_binary_model(
            train_df, label, epochs=args.epochs, save_dir=DATA_DIR
        )

    print("\nAll models trained and saved.")


if __name__ == "__main__":
    main()
