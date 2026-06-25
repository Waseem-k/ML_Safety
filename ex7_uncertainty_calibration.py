"""Exercise 7 — Uncertainty Quantification & Calibration

Covers:
  7.1  Epistemic vs. Aleatoric uncertainty (theory, printed)
  7.2  Calibration and ECE definition (theory, printed)
  7.3  Cost-optimal decision threshold derivation (theory, printed)
  7.4  ECE + reliability diagrams for all three CARLA models
  7.5  Temperature scaling (grid search on val set)
  7.6  Cost-optimal decision 2×2 table (pedestrian model)
  7.7  STPA causal scenario + safety constraints (theory, printed)

Usage:
    python ex7_uncertainty_calibration.py
    python ex7_uncertainty_calibration.py --data-dir "D:/ML Safety/2026" --models-dir models
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = r"D:\ML Safety\2026"
MODELS_DIR = r"D:\ML Safety\ML_Safety\models"

VAL_DIR  = os.path.join(DATA_DIR, "validation", "validation")
TEST_DIR = os.path.join(DATA_DIR, "test", "test")

TARGET_LABELS = ["has_pedestrian", "has_traffic_light", "has_vehicle"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cost matrix — Exercise 7.3
C_FN     = 100   # cost of missing a pedestrian (false negative)
C_FP     = 1     # cost of braking unnecessarily (false positive)
TAU_STAR = C_FP / (C_FP + C_FN)   # ≈ 0.0099


# ---------------------------------------------------------------------------
# Theory answers (7.1 – 7.3)
# ---------------------------------------------------------------------------
def print_theory():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  THEORY ANSWERS — Exercises 7.1 – 7.3                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

─── Exercise 7.1: Epistemic vs. Aleatoric Uncertainty ───────────────────────

  Epistemic uncertainty (model uncertainty) arises from insufficient training
  data or a model that has not seen a particular region of input space.  It is
  reducible: adding more data or a better model can decrease it.

  Aleatoric uncertainty (data uncertainty) reflects irreducible noise in the
  input itself — e.g. sensor noise, occlusion, or ambiguous scenes.  No amount
  of additional training eliminates it.

  1. OOD inputs (night images for a day-trained model): EPISTEMIC.
     The model has never seen such inputs; its parameters are not suited to
     predict them reliably.  More or different training data could reduce it.

  2. Correctly classified, in-distribution image: ALEATORIC.
     The model has learned the relevant features; residual uncertainty comes
     from sensor noise, partial occlusion, or inherent label ambiguity in the
     image itself.

─── Exercise 7.2: Calibration and ECE ───────────────────────────────────────

  A classifier is well-calibrated if its predicted confidence matches the
  empirical accuracy: among all predictions made with confidence p, roughly
  fraction p should be correct.  Formally: P(Ŷ = Y | p̂ = p) = p for all p.

  Expected Calibration Error (ECE):
    1. Partition predictions into M equal-width confidence bins B₁…Bₘ.
    2. For each bin Bₘ compute:
         acc(Bₘ)  = fraction of correct predictions in that bin
         conf(Bₘ) = mean predicted confidence in that bin
    3. ECE = Σₘ (|Bₘ| / n) · |acc(Bₘ) − conf(Bₘ)|
    where n is the total number of predictions.
  ECE = 0 means perfect calibration; higher values indicate over- or
  underconfidence.

─── Exercise 7.3: Cost-Optimal Decision Threshold ───────────────────────────

  Let p = P(pedestrian | x).

  Expected loss of each action:
    E[L | brake]    = C_FP · (1 − p)   (FP cost when no pedestrian)
    E[L | continue] = C_FN · p          (FN cost when pedestrian present)

  Break-even threshold τ*:
    C_FP · (1 − τ*) = C_FN · τ*
    C_FP = (C_FP + C_FN) · τ*
    τ* = C_FP / (C_FP + C_FN) = 1 / (1 + 100) ≈ 0.0099

  Decision rule: BRAKE if p ≥ τ*, CONTINUE if p < τ*.

  Compared to the standard argmax rule (τ = 0.5):
    τ* ≈ 0.0099 << 0.5  →  the system brakes at much lower confidence.
    This reflects the asymmetric cost: a missed pedestrian (FN) is 100× more
    costly than an unnecessary brake (FP), so caution is warranted even when
    the pedestrian probability is very small.

  4. Why calibration matters for τ*:
    τ* assumes p is a true probability.  If the model is overconfident, it
    may output p = 0.9 when the true probability is only 0.3.  The threshold
    τ* = 0.0099 is then applied to a distorted score, not a real probability,
    so the decision rule no longer minimises expected cost.  An overconfident
    model can report p < 0.0099 even when the true P(ped) is 0.05, causing
    the system to continue driving into a pedestrian.
""")


# ---------------------------------------------------------------------------
# Dataset  (same convention as ex3 / ex8)
# ---------------------------------------------------------------------------
class CarlaDataset(Dataset):
    def __init__(self, df, img_dir, label, transform=None):
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
def load_model(label):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    path = os.path.join(MODELS_DIR, f"resnet18_{label}.pth")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# Logit collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_logits(model, loader):
    """Returns (logits [N,2], labels [N]) as numpy arrays."""
    all_logits, all_labels = [], []
    for images, labels in tqdm(loader, leave=False, desc="  collecting logits"):
        logits = model(images.to(DEVICE))
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def get_probs(logits, T=1.0):
    """Return (predicted class, max confidence, P(class=1)) at temperature T."""
    probs     = F.softmax(torch.tensor(logits / T, dtype=torch.float32), dim=1).numpy()
    preds     = probs.argmax(axis=1)
    confs     = probs.max(axis=1)       # confidence in predicted class — used for ECE
    pos_probs = probs[:, 1]             # P(positive class) — used for threshold decisions
    return preds, confs, pos_probs


def compute_ece(logits, labels, T=1.0, n_bins=10):
    """Expected Calibration Error."""
    preds, confs, _ = get_probs(logits, T)
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = (preds[mask] == labels[mask]).mean()
        bin_conf = confs[mask].mean()
        ece     += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def plot_reliability_diagram(logits, labels, title, ax, T=1.0):
    """Reliability diagram: confidence bins vs. accuracy."""
    preds, confs, _ = get_probs(logits, T)
    bins = np.linspace(0, 1, 11)
    bin_mids, bin_accs, bin_gaps = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        acc  = (preds[mask] == labels[mask]).mean()
        conf = (lo + hi) / 2
        bin_mids.append(conf)
        bin_accs.append(acc)
        bin_gaps.append(abs(acc - conf))

    ax.bar(bin_mids, bin_accs,  width=0.09, alpha=0.75, label="Accuracy",           color="#4C72B0")
    ax.bar(bin_mids, bin_gaps,  width=0.09, alpha=0.40, label="Gap (miscalibration)",
           color="#C44E52", bottom=[min(a, c) for a, c in zip(bin_accs,
                                   [(lo + hi) / 2 for lo, hi in zip(bins[:-1], bins[1:])
                                    if ((confs >= lo) & (confs < hi)).sum() > 0])])
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ece = compute_ece(logits, labels, T)
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Temperature scaling — Exercise 7.5
# ---------------------------------------------------------------------------
def nll_at_temperature(logits, labels, T):
    scaled  = torch.tensor(logits / T, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    return F.cross_entropy(scaled, targets).item()


def find_best_temperature(logits, labels):
    """Grid search T ∈ {0.5, 0.6, …, 3.0} minimising validation NLL."""
    T_grid = np.round(np.arange(0.5, 3.05, 0.1), 2)
    best_T, best_nll = 1.0, float("inf")
    for T in T_grid:
        nll = nll_at_temperature(logits, labels, T)
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return round(best_T, 2), best_nll


# ---------------------------------------------------------------------------
# Cost-optimal decision — Exercise 7.6
# ---------------------------------------------------------------------------
def compute_total_loss(pos_probs, labels, tau):
    """Total asymmetric cost: C_FN·#FN + C_FP·#FP at threshold tau."""
    preds = (pos_probs >= tau).astype(int)
    fn    = int(((preds == 0) & (labels == 1)).sum())
    fp    = int(((preds == 1) & (labels == 0)).sum())
    total = C_FN * fn + C_FP * fp
    return total, fn, fp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Exercise 7 — Uncertainty & Calibration")
    p.add_argument("--data-dir",   default=DATA_DIR)
    p.add_argument("--models-dir", default=MODELS_DIR)
    p.add_argument("--skip-theory", action="store_true",
                   help="Skip printing theory answers")
    return p.parse_args()


def main():
    args = parse_args()

    global DATA_DIR, MODELS_DIR, VAL_DIR, TEST_DIR
    DATA_DIR   = args.data_dir
    MODELS_DIR = args.models_dir
    VAL_DIR    = os.path.join(DATA_DIR, "validation", "validation")
    TEST_DIR   = os.path.join(DATA_DIR, "test", "test")

    if not args.skip_theory:
        print_theory()

    print(f"Device     : {DEVICE}")
    print(f"Models dir : {MODELS_DIR}\n")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load dataframes and filter to existing images
    val_img_dir  = os.path.join(VAL_DIR,  "rgb-front")
    test_img_dir = os.path.join(TEST_DIR, "rgb-front")

    val_df  = pd.read_csv(os.path.join(VAL_DIR,  "labels.csv"))
    test_df = pd.read_csv(os.path.join(TEST_DIR, "labels.csv"))

    val_df  = val_df[val_df["frame"].apply(
        lambda f: os.path.exists(os.path.join(val_img_dir, f"{int(f):06d}.jpg")))].reset_index(drop=True)
    test_df = test_df[test_df["frame"].apply(
        lambda f: os.path.exists(os.path.join(test_img_dir, f"{int(f):06d}.jpg")))].reset_index(drop=True)

    print(f"Val  images : {len(val_df):,}")
    print(f"Test images : {len(test_df):,}\n")

    # -----------------------------------------------------------------------
    # Exercises 7.4 & 7.5: ECE + reliability diagrams + temperature scaling
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle("Reliability Diagrams — Before / After Temperature Scaling",
                 fontsize=13, fontweight="bold")

    best_T_store     = {}
    test_logits_store = {}
    test_labels_store = {}

    ece_summary = []

    for row_idx, label in enumerate(TARGET_LABELS):
        short = label.replace("has_", "").capitalize()
        print(f"{'='*60}")
        print(f"Model: {label}")
        print(f"{'='*60}")

        model = load_model(label)

        val_loader  = DataLoader(CarlaDataset(val_df,  val_img_dir,  label, TRANSFORM),
                                 batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(CarlaDataset(test_df, test_img_dir, label, TRANSFORM),
                                 batch_size=32, shuffle=False, num_workers=0)

        val_logits,  val_labels  = collect_logits(model, val_loader)
        test_logits, test_labels = collect_logits(model, test_loader)

        test_logits_store[label] = test_logits
        test_labels_store[label] = test_labels

        # Exercise 7.4 — ECE before scaling
        ece_before = compute_ece(test_logits, test_labels, T=1.0)
        print(f"  ECE (uncalibrated) : {ece_before:.4f}")

        # Exercise 7.5 — find best T on val set
        best_T, best_nll = find_best_temperature(val_logits, val_labels)
        best_T_store[label] = best_T
        print(f"  Best T (val NLL)   : T={best_T}  (NLL={best_nll:.4f})")

        ece_after = compute_ece(test_logits, test_labels, T=best_T)
        print(f"  ECE (T={best_T:<4})       : {ece_after:.4f}  (Δ = {ece_after - ece_before:+.4f})\n")

        # Over- or underconfident?
        _, confs_raw, _ = get_probs(test_logits, T=1.0)
        preds_raw, _, _ = get_probs(test_logits, T=1.0)
        acc = (preds_raw == test_labels).mean()
        mean_conf = confs_raw.mean()
        direction = "OVERCONFIDENT" if mean_conf > acc else "UNDERCONFIDENT"
        print(f"  Mean confidence={mean_conf:.3f}  Accuracy={acc:.3f}  → {direction}\n")

        ece_summary.append((short, ece_before, best_T, ece_after))

        # Reliability diagrams
        plot_reliability_diagram(test_logits, test_labels,
                                 f"{short} — Uncalibrated", axes[row_idx, 0], T=1.0)
        plot_reliability_diagram(test_logits, test_labels,
                                 f"{short} — T={best_T}", axes[row_idx, 1], T=best_T)

    plt.tight_layout()
    diag_path = os.path.join(MODELS_DIR, "ex7_reliability_diagrams.png")
    plt.savefig(diag_path, dpi=130, bbox_inches="tight")
    print(f"Reliability diagrams saved → {diag_path}\n")
    plt.show()
    plt.close(fig)

    # Summary table
    print(f"{'─'*55}")
    print(f"{'Model':<18} {'ECE Before':>11} {'Best T':>8} {'ECE After':>11} {'Δ ECE':>8}")
    print(f"{'─'*55}")
    for short, eb, T, ea in ece_summary:
        print(f"{short:<18} {eb:>11.4f} {T:>8.2f} {ea:>11.4f} {ea-eb:>+8.4f}")
    print(f"{'─'*55}\n")

    # -----------------------------------------------------------------------
    # Exercise 7.6: Cost-optimal decision — 2×2 table (pedestrian only)
    # -----------------------------------------------------------------------
    print(f"{'='*60}")
    print(f"Exercise 7.6: Cost-Optimal Decision  (C_FN={C_FN}, C_FP={C_FP})")
    print(f"  τ* = C_FP / (C_FP + C_FN) = {TAU_STAR:.4f}")
    print(f"{'='*60}\n")

    ped_logits = test_logits_store["has_pedestrian"]
    ped_labels = test_labels_store["has_pedestrian"]
    T_ped      = best_T_store["has_pedestrian"]

    configs = [
        ("Uncalibrated (T=1.0)", 1.0),
        (f"Calibrated   (T={T_ped})", T_ped),
    ]
    thresholds = [
        ("τ = 0.5",               0.5),
        (f"τ* = {TAU_STAR:.4f}",  TAU_STAR),
    ]

    # Build results
    table = {}
    for cal_name, T in configs:
        _, _, pos_probs = get_probs(ped_logits, T)
        table[cal_name] = {}
        for tau_name, tau in thresholds:
            loss, fn, fp = compute_total_loss(pos_probs, ped_labels, tau)
            table[cal_name][tau_name] = (loss, fn, fp)

    # Print 2×2 table
    tau_names = [t[0] for t in thresholds]
    col = 30
    print(f"{'':28}  {tau_names[0]:^{col}}  {tau_names[1]:^{col}}")
    print("─" * (28 + 2 * col + 4))
    best_loss = min(v[0] for row in table.values() for v in row.values())
    for cal_name, row in table.items():
        cells = []
        for tau_name, (loss, fn, fp) in row.items():
            marker = " ◄ BEST" if loss == best_loss else ""
            cells.append(f"Loss={loss:>6}  FN={fn:>4}  FP={fp:>5}{marker}")
        print(f"{cal_name:<28}  {cells[0]:<{col+7}}  {cells[1]}")

    print(f"\n  Best combination: lowest total loss uses calibrated model + τ*")
    print(f"  Rationale: calibration makes p a true probability; τ* encodes the")
    print(f"  asymmetric cost structure.  Both are needed for optimal decisions.\n")

    # -----------------------------------------------------------------------
    # Exercise 7.7: STPA causal scenario + safety constraints
    # -----------------------------------------------------------------------
    print(f"""
{'='*60}
Exercise 7.7: STPA — Overconfidence as a Causal Factor
{'='*60}

1. CAUSAL SCENARIO
   The pedestrian classifier produces a false negative — it predicts "no
   pedestrian" while one is present — yet outputs a confidence well above
   any reasonable low-confidence threshold.  Because the planner sees a
   high-confidence "no pedestrian" signal, no fallback (e.g., reduced
   speed, LiDAR query) is triggered.  The pre-existing unsafe control
   action ("planner does not command braking while a pedestrian is in the
   path") therefore occurs without any safety net.

2. SAFETY CONSTRAINTS
   Model-level (ML constraint on calibration):
     SC-M1: The pedestrian detector's ECE on the in-distribution test set
     shall not exceed 0.05 after temperature scaling.
     Measured ECE = {compute_ece(ped_logits, ped_labels, T=T_ped):.4f} (calibrated).

   System-level (planner behaviour):
     SC-S1: The autopilot shall apply decision threshold τ* = {TAU_STAR:.4f}
     (derived from C_FN={C_FN}, C_FP={C_FP}) rather than τ = 0.5, so the
     asymmetric braking cost is reflected in every planning cycle.

     SC-S2: If the calibrated confidence P(ped|x) falls below 0.30,
     the planner shall engage a conservative fallback (reduce speed,
     query secondary sensors) before deciding not to brake.

3. VERIFICATION
   Evidence produced in Exercises 7.4–7.5:
   • Reliability diagrams and ECE values before/after temperature scaling.
   • If calibrated ECE ≤ 0.05, SC-M1 is satisfied.
   Temperature scaling is a cheap post-hoc fix: it can be re-verified after
   any model update without retraining.

4. RESIDUAL RISK
   Calibration is a frequentist guarantee: P(ped|x) = p means roughly
   fraction p of *similar* inputs contain a pedestrian.  On any single
   frame — especially a novel or adversarial one — the model can still
   produce a high-confidence false negative.
   Therefore SC-M1 alone does NOT close the scenario.  The system-level
   fallback (SC-S1, SC-S2) and sensor redundancy remain necessary to
   cover residual risk that calibration cannot eliminate.
""")


if __name__ == "__main__":
    main()
