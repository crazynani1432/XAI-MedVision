import os
import sys
import time
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import classification_report, confusion_matrix

def find_blood_cell_dirs(base_path):
    train_dir, test_dir = None, None
    for root, dirs, _ in os.walk(base_path):
        if 'TRAIN' in dirs and 'TEST' in dirs:
            # Check if TRAIN directory contains class subfolders
            candidate_train = os.path.join(root, 'TRAIN')
            candidate_test = os.path.join(root, 'TEST')
            subdirs = os.listdir(candidate_train)
            if any(c in subdirs for c in ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']):
                train_dir = candidate_train
                test_dir = candidate_test
                break
    if not train_dir or not test_dir:
        for root, dirs, _ in os.walk(base_path):
            if os.path.basename(root).upper() == 'TRAIN' and os.path.isdir(root):
                train_dir = root
            if os.path.basename(root).upper() == 'TEST' and os.path.isdir(root):
                test_dir = root
    return train_dir, test_dir

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("=========================================================================", flush=True)
    print("--- Phase 4: Hematology CBC Blood Cell MobileNet-V2 Training (Target >95%) ---", flush=True)
    print("=========================================================================", flush=True)

    start_time_global = time.time()

    # 1. Download Dataset
    print("[1/5] Downloading Blood Cell dataset via kagglehub...", flush=True)
    dataset_path = kagglehub.dataset_download("paultimothymooney/blood-cells")
    print(f"Dataset root: {dataset_path}", flush=True)

    train_dir, test_dir = find_blood_cell_dirs(dataset_path)
    if not train_dir or not test_dir:
        raise FileNotFoundError(f"Could not locate TRAIN/TEST directories in {dataset_path}")

    print(f"Found train directory: {train_dir}", flush=True)
    print(f"Found test directory: {test_dir}", flush=True)

    # 2. Transformations & Data Loaders
    print("[2/5] Initializing microscopic image augmentation pipeline...", flush=True)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

    classes = train_dataset.classes
    print(f"Classes identified ({len(classes)}): {classes}", flush=True)
    print(f"Class mapping: {train_dataset.class_to_idx}", flush=True)
    print(f"Train dataset size: {len(train_dataset)} | Test dataset size: {len(test_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 3. Model Architecture (MobileNetV2 Fine-tuning)
    print("[3/5] Setting up MobileNet-V2 Transfer Learning Architecture...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution hardware device: {device}", flush=True)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze earlier layers, unfreeze feature layers 14+ for cell histology domain adaptation
    for param in model.features[:14].parameters():
        param.requires_grad = False
    for param in model.features[14:].parameters():
        param.requires_grad = True

    model.classifier[1] = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(classes))
    )

    os.makedirs("models_checkpoints", exist_ok=True)
    checkpoint_path = "models_checkpoints/blood_cell_mobilenet.pth"

    best_acc = 0.0
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}. Attempting weight restoration...", flush=True)
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Successfully restored existing checkpoint!", flush=True)
        except Exception as e:
            print(f"Checkpoint format update required: {e}. Starting clean fine-tuning...", flush=True)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Evaluate baseline accuracy if loaded checkpoint exists
    if os.path.exists(checkpoint_path):
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)
        best_acc = correct_val / total_val if total_val > 0 else 0.0
        print(f"Loaded Baseline Checkpoint Accuracy: {best_acc*100:.2f}%", flush=True)

    # 4. Training Loop with Live Status & Time Estimates
    print(f"[4/5] Starting 10-Epoch Optimization Loop (Target Accuracy >95%)...", flush=True)
    
    epoch_times = []
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        batch_idx = 0
        total_batches = len(train_loader)
        
        for inputs, labels in train_loader:
            batch_idx += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += inputs.size(0)

            if batch_idx % 40 == 0 or batch_idx == total_batches:
                curr_acc = (correct_train / total_train) * 100
                print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx}/{total_batches} | Train Loss: {loss.item():.4f} | Acc: {curr_acc:.2f}%", flush=True)

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation Step
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch
        est_remaining_sec = avg_epoch_time * remaining_epochs
        est_remaining_min = est_remaining_sec / 60.0

        print(f"==> Epoch {epoch}/{epochs} Summary:")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"    Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc*100:.2f}%")
        print(f"    LR: {current_lr:.6f} | Epoch Time: {epoch_duration:.1f}s | Est. Remaining: {est_remaining_min:.1f} min", flush=True)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    [+] NEW BEST MODEL SAVED! Validation Accuracy: {best_acc*100:.2f}%", flush=True)

    print("=========================================================================", flush=True)
    print(f"Training Complete! Peak Validation Accuracy: {best_acc*100:.2f}%", flush=True)
    print("=========================================================================", flush=True)

    # 5. Final Metrics & Grad-CAM Visualization
    print("[5/5] Computing final classification metrics & Grad-CAM visualization...", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Classification Report
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    print("\n--- Detailed Classification Metrics ---", flush=True)
    print(classification_report(all_targets, all_preds, target_names=classes), flush=True)

    # Grad-CAM Heatmap
    sample_img_path = None
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                sample_img_path = os.path.join(root, f)
                break
        if sample_img_path:
            break

    if sample_img_path:
        print(f"Generating Grad-CAM visualization for sample image: {sample_img_path}", flush=True)
        pil_img = Image.open(sample_img_path).convert('RGB')
        pil_img_resized = pil_img.resize((224, 224))
        rgb_img = np.float32(pil_img_resized) / 255.0

        input_tensor = val_transforms(pil_img).unsqueeze(0).to(device)

        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()

        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        os.makedirs('static', exist_ok=True)
        gradcam_output_path = 'static/sample_blood_gradcam.png'
        Image.fromarray(visualization).save(gradcam_output_path)
        print(f"Saved Grad-CAM heatmap overlay to {gradcam_output_path}", flush=True)

    total_elapsed = (time.time() - start_time_global) / 60.0
    print(f"Total Phase 4 Execution Time: {total_elapsed:.2f} minutes", flush=True)

if __name__ == '__main__':
    main()
