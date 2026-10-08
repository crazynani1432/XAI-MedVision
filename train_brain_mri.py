"""
XAI-MedVision -- Phase 1: Brain Tumor MRI Classifier
=====================================================
Downloads the brain-tumor-mri-dataset from Kaggle, fine-tunes EfficientNet-B0,
evaluates on the test split, and generates a Grad-CAM heatmap for visual
explainability. Supports checkpoint resumption and configurable training epochs.
"""

import os
import sys
import copy
import pathlib
import argparse

# pyrefly: ignore [missing-import]
import kagglehub
# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────
# 0.  Configuration
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
EPOCHS = 10
BATCH_SIZE = 32
LR = 2e-4
IMAGE_SIZE = 224

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "models_checkpoints"
STATIC_DIR = PROJECT_ROOT / "static"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "brain_mri_efficientnet.pth"
GRADCAM_PATH = STATIC_DIR / "sample_gradcam.png"


# ──────────────────────────────────────────────
# 1.  Download dataset via kagglehub
# ──────────────────────────────────────────────
def download_dataset():
    """Download the brain-tumor-mri-dataset and return Training/Testing paths."""
    print("[*] Downloading / locating dataset from Kaggle ...", flush=True)
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    dataset_path = pathlib.Path(dataset_path)
    print(f"[OK] Dataset root: {dataset_path}", flush=True)

    train_dir = None
    test_dir = None
    for root, dirs, _ in os.walk(dataset_path):
        root_p = pathlib.Path(root)
        if "Training" in dirs:
            train_dir = root_p / "Training"
        if "Testing" in dirs:
            test_dir = root_p / "Testing"
        if train_dir and test_dir:
            break

    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            f"Could not locate Training/Testing directories under {dataset_path}"
        )

    print(f"   Training dir : {train_dir}", flush=True)
    print(f"   Testing  dir : {test_dir}", flush=True)
    return train_dir, test_dir


# ──────────────────────────────────────────────
# 2.  Transforms & DataLoaders
# ──────────────────────────────────────────────
def build_dataloaders(train_dir, test_dir, batch_size=BATCH_SIZE):
    """Build augmented training and normalized validation DataLoaders."""
    train_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=val_transforms)

    print(f"   Classes (train): {train_dataset.class_to_idx}", flush=True)
    print(f"   Classes (test) : {test_dataset.class_to_idx}", flush=True)

    # Use num_workers=0 on Windows for reliable CPU execution without spawn overhead
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader, train_dataset.class_to_idx


# ──────────────────────────────────────────────
# 3.  Model construction
# ──────────────────────────────────────────────
def build_model(resume=True):
    """Instantiate EfficientNet-B0. If a saved checkpoint exists and resume=True, load it."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    if resume and CHECKPOINT_PATH.exists():
        print(f"[*] Resuming from existing checkpoint: {CHECKPOINT_PATH}", flush=True)
        state_dict = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print("[*] Initialized model with default pretrained ImageNet weights.", flush=True)

    return model.to(DEVICE)


# ──────────────────────────────────────────────
# 4.  Training loop
# ──────────────────────────────────────────────
def train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LR, start_epoch=1):
    """Train for specified epochs, saving the best weights by validation accuracy."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Advance scheduler if continuing from a later epoch
    for _ in range(1, start_epoch):
        scheduler.step()

    # Calculate baseline validation accuracy
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    best_acc = val_correct / val_total if val_total > 0 else 0.0
    print(f"[*] Starting baseline validation accuracy: {best_acc:.4f}", flush=True)
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, epochs + 1):
        # -- Train phase --
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            if batch_idx % 20 == 0 or batch_idx == len(train_loader):
                print(
                    f"  [Epoch {epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} "
                    f"-- loss: {loss.item():.4f}",
                    flush=True,
                )

        train_loss = running_loss / total
        train_acc = correct / total

        # -- Validation phase --
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{epochs}  > lr={current_lr:.6f}  train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}",
            flush=True,
        )
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, str(CHECKPOINT_PATH))
            print(f"  * New best val_acc={val_acc:.4f} -- checkpoint saved to {CHECKPOINT_PATH}", flush=True)

    # Reload best weights
    model.load_state_dict(best_weights)
    torch.save(best_weights, str(CHECKPOINT_PATH))
    print(f"\n[OK] Training complete. Best val_acc = {best_acc:.4f}", flush=True)
    return model


# ──────────────────────────────────────────────
# 5.  Evaluation
# ──────────────────────────────────────────────
def evaluate_model(model, test_loader):
    """Print a full classification report on the test set."""
    print("\n[*] Running test evaluation ...", flush=True)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "=" * 60, flush=True)
    print("Classification Report", flush=True)
    print("=" * 60, flush=True)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES), flush=True)
    print("Confusion Matrix:", flush=True)
    print(confusion_matrix(all_labels, all_preds), flush=True)


# ──────────────────────────────────────────────
# 6.  Grad-CAM heatmap generation
# ──────────────────────────────────────────────
def generate_gradcam(model, test_dir):
    """Generate a Grad-CAM overlay from a glioma test sample and save to static/."""
    glioma_dir = pathlib.Path(test_dir) / "glioma"
    sample_path = next(glioma_dir.iterdir())
    print(f"\n[*] Generating Grad-CAM for: {sample_path.name}", flush=True)

    raw_image = Image.open(sample_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_array = np.array(raw_image, dtype=np.float32) / 255.0

    preprocess = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(raw_image).unsqueeze(0).to(DEVICE)

    target_layer = [model.features[-1]]
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]

    overlay = show_cam_on_image(rgb_array, grayscale_cam, use_rgb=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_array)
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(GRADCAM_PATH), dpi=150)
    plt.close()
    print(f"[OK] Grad-CAM saved -> {GRADCAM_PATH}", flush=True)


# ──────────────────────────────────────────────
# 7.  Main entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="XAI-MedVision Phase 1: Train & Evaluate Brain MRI Classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Total target epochs (default: {EPOCHS})")
    parser.add_argument("--start-epoch", type=int, default=4, help="Starting epoch number when continuing training (default: 4)")
    parser.add_argument("--lr", type=float, default=LR, help=f"Learning rate (default: {LR})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size (default: 32)")
    parser.add_argument("--no-resume", action="store_true", help="Start training from scratch, ignoring existing checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation and Grad-CAM generation")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("  XAI-MedVision -- Phase 1: Brain Tumor MRI Classifier", flush=True)
    print("=" * 60, flush=True)
    print(f"Device : {DEVICE}", flush=True)

    train_dir, test_dir = download_dataset()
    train_loader, test_loader, class_to_idx = build_dataloaders(train_dir, test_dir, batch_size=args.batch_size)

    resume = not args.no_resume
    model = build_model(resume=resume)
    print(f"\n[*] Model: EfficientNet-B0 | Classifier head -> {NUM_CLASSES} classes", flush=True)

    if not args.eval_only:
        model = train_model(
            model,
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            start_epoch=args.start_epoch,
        )

    evaluate_model(model, test_loader)
    generate_gradcam(model, test_dir)

    print("\n[DONE] Phase 1 pipeline executed successfully!", flush=True)


if __name__ == "__main__":
    main()
