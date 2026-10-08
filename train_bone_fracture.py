"""
XAI-MedVision -- Phase 3: Bone Fracture X-Ray Classifier (Model 3)
==================================================================
Downloads the fracture-multi-region-x-ray-data from Kaggle, fine-tunes ResNet-50
(layer3, layer4 + custom classification head) with high-generalization augmentations,
evaluates on validation & test sets, and generates a Grad-CAM heatmap overlay.
Saves the best model weights to models_checkpoints/bone_fracture_resnet50.pth.
"""

import os
import sys
import copy
import pathlib
import argparse

import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────
# 0. Configuration
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
NUM_CLASSES = 2
CLASS_NAMES = ["fractured", "not fractured"]
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "models_checkpoints"
STATIC_DIR = PROJECT_ROOT / "static"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "bone_fracture_resnet50.pth"
GRADCAM_PATH = STATIC_DIR / "sample_bone_gradcam.png"


# ──────────────────────────────────────────────
# 1. Download dataset via kagglehub
# ──────────────────────────────────────────────
def download_dataset():
    """Download bone fracture X-ray dataset and return train, val, and test paths."""
    print("[*] Downloading / locating bone fracture dataset from Kaggle ...", flush=True)
    dataset_path = kagglehub.dataset_download("bmadushanirodrigo/fracture-multi-region-x-ray-data")
    dataset_path = pathlib.Path(dataset_path)
    print(f"[OK] Dataset root: {dataset_path}", flush=True)

    train_dir, val_dir, test_dir = None, None, None
    for root, dirs, _ in os.walk(dataset_path):
        root_p = pathlib.Path(root)
        if "train" in dirs and "val" in dirs and "test" in dirs:
            train_dir = root_p / "train"
            val_dir = root_p / "val"
            test_dir = root_p / "test"
            break
        elif "train" in dirs and "test" in dirs:
            train_dir = root_p / "train"
            test_dir = root_p / "test"

    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            f"Could not locate train/test directories under {dataset_path}"
        )
    if val_dir is None:
        val_dir = test_dir

    print(f"   Train dir : {train_dir}", flush=True)
    print(f"   Val   dir : {val_dir}", flush=True)
    print(f"   Test  dir : {test_dir}", flush=True)
    return train_dir, val_dir, test_dir


# ──────────────────────────────────────────────
# 2. Transforms & DataLoaders
# ──────────────────────────────────────────────
def build_dataloaders(train_dir, val_dir, test_dir, batch_size=BATCH_SIZE):
    """Build augmented training and normalized val/test DataLoaders."""
    train_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transforms = transforms.Compose(
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
    val_dataset = datasets.ImageFolder(str(val_dir), transform=eval_transforms)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_transforms)

    print(f"   Classes (train): {train_dataset.class_to_idx}", flush=True)
    print(f"   Total Train images: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


# ──────────────────────────────────────────────
# 3. Model construction (ResNet-50 Transfer Learning)
# ──────────────────────────────────────────────
def build_model(resume=True):
    """
    Instantiate ResNet-50.
    Unfreeze layer3 and layer4, freeze earlier layers for stability & speed.
    Custom classification head: Linear(2048->256) -> ReLU -> Dropout(0.3) -> Linear(256->2).
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze conv1, bn1, layer1, layer2
    for name, param in model.named_parameters():
        if not ("layer3" in name or "layer4" in name or "fc" in name):
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )

    if resume and CHECKPOINT_PATH.exists():
        print(f"[*] Resuming from existing checkpoint: {CHECKPOINT_PATH}", flush=True)
        state_dict = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print("[*] Initialized ResNet-50 with pretrained ImageNet weights.", flush=True)

    return model.to(DEVICE)


# ──────────────────────────────────────────────
# 4. Training loop
# ──────────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, start_epoch=1):
    """Train for specified epochs, saving the best weights by validation accuracy."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # Baseline validation accuracy
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    best_acc = val_correct / val_total if val_total > 0 else 0.0
    print(f"[*] Starting baseline validation accuracy: {best_acc:.4f}", flush=True)
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, epochs + 1):
        # -- Train Phase --
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            if batch_idx % 40 == 0 or batch_idx == len(train_loader):
                print(
                    f"  [Epoch {epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} "
                    f"-- loss: {loss.item():.4f}",
                    flush=True,
                )

        train_loss = running_loss / total
        train_acc = correct / total

        # -- Validation Phase --
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
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
# 5. Evaluation
# ──────────────────────────────────────────────
def evaluate_model(model, test_loader):
    """Print classification report and confusion matrix on the test set."""
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
    print("Bone Fracture Classification Report", flush=True)
    print("=" * 60, flush=True)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES), flush=True)
    print("Confusion Matrix:", flush=True)
    print(confusion_matrix(all_labels, all_preds), flush=True)


# ──────────────────────────────────────────────
# 6. Grad-CAM Heatmap Generation
# ──────────────────────────────────────────────
def generate_gradcam(model, test_dir):
    """Generate a Grad-CAM overlay from a fractured X-ray sample and save to static/."""
    fractured_dir = pathlib.Path(test_dir) / "fractured"
    sample_path = next(fractured_dir.iterdir())
    print(f"\n[*] Generating Bone Fracture Grad-CAM for: {sample_path.name}", flush=True)

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

    target_layer = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]

    overlay = show_cam_on_image(rgb_array, grayscale_cam, use_rgb=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_array)
    axes[0].set_title("Original X-Ray")
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
    print(f"[OK] Bone Fracture Grad-CAM saved -> {GRADCAM_PATH}", flush=True)


# ──────────────────────────────────────────────
# 7. Main Entry Point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="XAI-MedVision Phase 3: Train & Evaluate Bone Fracture Classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Total target epochs (default: {EPOCHS})")
    parser.add_argument("--start-epoch", type=int, default=9, help="Starting epoch when continuing training (default: 9)")
    parser.add_argument("--lr", type=float, default=LR, help=f"Learning rate (default: {LR})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--no-resume", action="store_true", help="Start training from scratch, ignoring existing checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation and Grad-CAM generation")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("  XAI-MedVision -- Phase 3: Bone Fracture Classifier", flush=True)
    print("=" * 60, flush=True)
    print(f"Device : {DEVICE} | AMP : {USE_AMP}", flush=True)

    train_dir, val_dir, test_dir = download_dataset()
    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        train_dir, val_dir, test_dir, batch_size=args.batch_size
    )

    resume = not args.no_resume
    model = build_model(resume=resume)
    print(f"\n[*] Model: ResNet-50 | Head -> {NUM_CLASSES} classes ({CLASS_NAMES})", flush=True)

    if not args.eval_only:
        model = train_model(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.lr,
            start_epoch=args.start_epoch,
        )

    evaluate_model(model, test_loader)
    generate_gradcam(model, test_dir)

    print("\n[DONE] Phase 3 pipeline executed successfully!", flush=True)


if __name__ == "__main__":
    main()
