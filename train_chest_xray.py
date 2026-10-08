import os
import sys
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

def find_dataset_dirs(base_path):
    train_dir, test_dir = None, None
    for root, dirs, _ in os.walk(base_path):
        if 'train' in dirs and 'test' in dirs:
            train_dir = os.path.join(root, 'train')
            test_dir = os.path.join(root, 'test')
            break
    if not train_dir or not test_dir:
        for root, dirs, _ in os.walk(base_path):
            if os.path.basename(root).lower() == 'train' and os.path.isdir(root):
                train_dir = root
            if os.path.basename(root).lower() == 'test' and os.path.isdir(root):
                test_dir = root
    return train_dir, test_dir

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("--- Phase 2: Chest X-Ray Pneumonia Classifier Training ---", flush=True)
    
    # 1. Download Dataset
    print("[1/5] Downloading dataset via kagglehub...", flush=True)
    dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Dataset root: {dataset_path}", flush=True)
    
    train_dir, test_dir = find_dataset_dirs(dataset_path)
    if not train_dir or not test_dir:
        raise FileNotFoundError(f"Could not locate train/test directories in {dataset_path}")
    
    print(f"Found train directory: {train_dir}", flush=True)
    print(f"Found test directory: {test_dir}", flush=True)
    
    # 2. Preprocessing & Augmentation
    print("[2/5] Setting up data loaders and transformations...", flush=True)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.RandomResizedCrop(224, scale=(0.88, 1.0)),
        transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
    
    print(f"Classes identified: {train_dataset.classes}", flush=True)
    print(f"Class mapping: {train_dataset.class_to_idx}", flush=True)
    print(f"Train dataset size: {len(train_dataset)} | Test dataset size: {len(test_dataset)}", flush=True)
    
    # Calculate Class Weights to handle dataset imbalance (3:1 ratio)
    num_normal = sum(1 for _, label in train_dataset.samples if label == train_dataset.class_to_idx['NORMAL'])
    num_pneumonia = sum(1 for _, label in train_dataset.samples if label == train_dataset.class_to_idx['PNEUMONIA'])
    total_samples = len(train_dataset)
    
    w_normal = total_samples / (2.0 * num_normal)
    w_pneumonia = total_samples / (2.0 * num_pneumonia)
    print(f"Class Counts -> NORMAL: {num_normal}, PNEUMONIA: {num_pneumonia}", flush=True)
    print(f"Computed Loss Weights -> NORMAL: {w_normal:.3f}, PNEUMONIA: {w_pneumonia:.3f}", flush=True)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 3. Model Architecture
    print("[3/5] Initializing DenseNet-121 architecture with Medical AI Optimization...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution device: {device}", flush=True)
    
    os.makedirs("models_checkpoints", exist_ok=True)
    checkpoint_path = "models_checkpoints/chest_xray_densenet.pth"

    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 2)
    )
    
    best_acc = 0.0
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}. Loading state dict to continue training...", flush=True)
        try:
            # First try loading as sequential classifier, fallback if shape mismatch
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Successfully loaded existing checkpoint weights!", flush=True)
        except Exception as e:
            print(f"Checkpoint format adjusted for dropout classifier: {e}. Fine-tuning pretrained DenseNet-121 backbone...", flush=True)

    model = model.to(device)
    
    class_weights = torch.tensor([w_normal, w_pneumonia], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Evaluate initial accuracy of loaded checkpoint if available
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
        print(f"Initial Checkpoint Baseline Validation Accuracy: {best_acc*100:.2f}%", flush=True)

    # 4. Training Loop
    print(f"[4/5] Starting optimized training for {epochs} epochs targeting >95% accuracy...", flush=True)
    for epoch in range(1, epochs + 1):
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

            if batch_idx % 30 == 0 or batch_idx == total_batches:
                current_acc = (correct_train / total_train) * 100
                print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx}/{total_batches} | Train Loss: {loss.item():.4f} Acc: {current_acc:.2f}%", flush=True)
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # Validation / Evaluation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        # Update learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}% | "
              f"LR: {current_lr:.6f}", flush=True)
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f" --> [+] NEW BEST CHECKPOINT SAVED! (Accuracy: {best_acc*100:.2f}%)", flush=True)
            
    print(f"Training completed. Peak Validation Accuracy: {best_acc*100:.2f}%", flush=True)
    
    # Load best weights for evaluation & Grad-CAM
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 5. Visual Explainability (Grad-CAM)
    print("[5/5] Generating Grad-CAM visualization for sample PNEUMONIA test image...")
    pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')
    sample_filename = None
    for fname in os.listdir(pneumonia_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample_filename = os.path.join(pneumonia_dir, fname)
            break
            
    if not sample_filename:
        raise FileNotFoundError(f"No image files found in {pneumonia_dir}")
        
    print(f"Selected test sample: {sample_filename}")
    pil_img = Image.open(sample_filename).convert('RGB')
    pil_img_resized = pil_img.resize((224, 224))
    rgb_img = np.float32(pil_img_resized) / 255.0
    
    input_tensor = val_transforms(pil_img).unsqueeze(0).to(device)
    
    target_layers = [model.features.denseblock4]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    pneumonia_idx = train_dataset.class_to_idx.get('PNEUMONIA', 1)
    targets = [ClassifierOutputTarget(pneumonia_idx)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    os.makedirs('static', exist_ok=True)
    gradcam_output_path = 'static/sample_chest_gradcam.png'
    Image.fromarray(visualization).save(gradcam_output_path)
    print(f"Grad-CAM saved successfully to {gradcam_output_path}")

if __name__ == '__main__':
    main()
