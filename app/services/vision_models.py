import os
import io
import gc
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Try loading pytorch_grad_cam; if system policy blocks sklearn DLLs, use PyTorch native GradCAM
_USE_PYTORCH_GRAD_CAM = False
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _USE_PYTORCH_GRAD_CAM = True
    print("[GradCAM] Using pytorch_grad_cam package.")
except Exception as e:
    print(f"[GradCAM Warning] Falling back to PyTorch Native GradCAM: {e}")

_active_model = None
_active_model_name = None
_device = None

def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

def unload_other_models(current_name):
    global _active_model, _active_model_name, _chest_model, _blood_model, _brain_model, _bone_model
    if _active_model_name != current_name:
        _chest_model = None
        _blood_model = None
        _brain_model = None
        _bone_model = None
        _active_model = None
        gc.collect()
        _active_model_name = current_name

def overlay_heatmap_on_image(rgb_img, grayscale_cam):
    """Blends a grayscale CAM heatmap (0..1) with an RGB image (0..1) using Jet colormap."""
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    cam_img = 0.5 * heatmap + 0.5 * rgb_img
    cam_img = cam_img / np.max(cam_img)
    return np.uint8(255 * cam_img)

def compute_native_gradcam(model, target_layer, input_tensor, pred_idx):
    """Computes Grad-CAM using native PyTorch hooks and tensor operations."""
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    outputs = model(input_tensor)
    score = outputs[0, pred_idx]
    score.backward(retain_graph=True)

    h1.remove()
    h2.remove()

    if not features or not gradients:
        return np.zeros((224, 224), dtype=np.float32)

    feat = features[0].detach()
    grad = gradients[0].detach()

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feat, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros_like(cam)

    return cam

def validate_scan_domain(pil_img_resized, modality):
    """Clinical Domain Screener: Validates medical modality features and rejects non-medical photos."""
    arr = np.array(pil_img_resized, dtype=np.float32)
    sub = arr[::4, ::4, :]
    r, g, b = sub[:,:,0], sub[:,:,1], sub[:,:,2]
    maxC = np.maximum(np.maximum(r, g), b)
    minC = np.minimum(np.minimum(r, g), b)
    maxC_safe = np.maximum(maxC, 1e-5)
    sat = np.where(maxC > 0, (maxC - minC) / maxC_safe, 0)
    
    avg_sat = float(np.mean(sat))
    avg_rg = float(np.mean(np.abs(r - g)))
    avg_gb = float(np.mean(np.abs(g - b)))
    is_mono = (avg_rg < 8.0 and avg_gb < 8.0 and avg_sat < 0.10)
    
    backlight = float(np.mean(maxC > 150))
    purple = float(np.mean((r > g * 1.15) & (b > g * 1.05) & (r > 50) & (b > 50)))
    skin = float(np.mean((r > g) & (g > b) & ((r - g) > 12) & ((g - b) > 6) & (sat > 0.15)))
    
    if modality == 'blood':
        if is_mono or skin >= 0.05 or purple < 0.012 or backlight < 0.30:
            return False, "Non-hematology image detected. Blood cell smears require Giemsa cytochemical stain and illuminated microscope condenser backlight."
        return True, ""
    else:
        # Radiographs (Chest X-Ray, Brain MRI, Bone Radiograph) MUST be monochrome
        if not is_mono:
            return False, "Non-radiological color photograph detected. Clinical radiographs (Chest X-Ray, Brain MRI, Bone Radiograph) are monochrome greyscale scans."
        return True, ""

# ----------------------------------------------------
# 1. Chest X-Ray Pneumonia Model
# ----------------------------------------------------
def get_chest_model(model_path="models_checkpoints/chest_xray_densenet.pth"):
    global _active_model
    unload_other_models('chest')
    device = get_device()
    if _active_model is None:
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier.in_features, 2)
        )
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"[Model Loaded] Successfully restored Chest X-Ray weights from {model_path}")
            except Exception as e:
                print(f"[Model Warning] Failed loading {model_path}: {e}")
        else:
            print(f"[Model Warning] Checkpoint {model_path} not found. Using pretrained DenseNet-121 weights.")
            
        model.to(device)
        model.eval()
        _active_model = model
    return _active_model, device

def predict_chest_xray(image_bytes, model_path="models_checkpoints/chest_xray_densenet.pth"):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    pil_img_resized = pil_img.resize((224, 224))
    
    is_valid, reason = validate_scan_domain(pil_img_resized, 'xray')
    if not is_valid:
        return {
            "label": "CANNOT BE DIAGNOSED",
            "confidence": 0.0,
            "class_probabilities": {},
            "gradcam_base64": None,
            "is_valid": False,
            "reason": reason
        }

    model, device = get_chest_model(model_path)
    rgb_img = np.float32(pil_img_resized) / 255.0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    classes = ['NORMAL', 'PNEUMONIA']
    label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)

    target_layer = model.features.denseblock4
    if _USE_PYTORCH_GRAD_CAM:
        try:
            cam = GradCAM(model=model, target_layers=[target_layer])
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        except Exception:
            grayscale_cam = compute_native_gradcam(model, target_layer, input_tensor, pred_idx)
            visualization = overlay_heatmap_on_image(rgb_img, grayscale_cam)
    else:
        grayscale_cam = compute_native_gradcam(model, target_layer, input_tensor, pred_idx)
        visualization = overlay_heatmap_on_image(rgb_img, grayscale_cam)

    buffered = io.BytesIO()
    Image.fromarray(visualization).save(buffered, format="PNG")
    gradcam_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "label": label,
        "confidence": float(confidence),
        "gradcam_base64": gradcam_b64
    }

# ----------------------------------------------------
# 2. Hematology Blood Cell (CBC) Model
# ----------------------------------------------------
def get_blood_model(model_path="models_checkpoints/blood_cell_mobilenet.pth"):
    global _active_model
    unload_other_models('blood')
    device = get_device()
    if _active_model is None:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"[Model Loaded] Successfully restored Blood Cell weights from {model_path}")
            except Exception as e:
                print(f"[Model Warning] Failed loading {model_path}: {e}")
        else:
            print(f"[Model Warning] Checkpoint {model_path} not found. Using pretrained MobileNetV2 weights.")

        model.to(device)
        model.eval()
        _active_model = model
    return _active_model, device

def predict_blood_cell(image_bytes, model_path="models_checkpoints/blood_cell_mobilenet.pth"):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    pil_img_resized = pil_img.resize((224, 224))
    
    is_valid, reason = validate_scan_domain(pil_img_resized, 'blood')
    if not is_valid:
        return {
            "label": "CANNOT BE DIAGNOSED",
            "confidence": 0.0,
            "class_probabilities": {},
            "gradcam_base64": None,
            "is_valid": False,
            "reason": reason
        }

    model, device = get_blood_model(model_path)
    rgb_img = np.float32(pil_img_resized) / 255.0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)

    target_layer = model.features[-1]
    if _USE_PYTORCH_GRAD_CAM:
        try:
            cam = GradCAM(model=model, target_layers=[target_layer])
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        except Exception:
            grayscale_cam = compute_native_gradcam(model, target_layer, input_tensor, pred_idx)
            visualization = overlay_heatmap_on_image(rgb_img, grayscale_cam)
    else:
        grayscale_cam = compute_native_gradcam(model, target_layer, input_tensor, pred_idx)
        visualization = overlay_heatmap_on_image(rgb_img, grayscale_cam)

    buffered = io.BytesIO()
    Image.fromarray(visualization).save(buffered, format="PNG")
    gradcam_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    prob_dict = {classes[i]: round(float(probs[i]) * 100, 2) for i in range(len(classes))}

    return {
        "label": label,
        "confidence": float(confidence),
        "class_probabilities": prob_dict,
        "gradcam_base64": gradcam_b64
    }

# ----------------------------------------------------
# 3. Brain Tumor MRI Model (EfficientNet-B0)
# ----------------------------------------------------
def get_brain_model(model_path="models_checkpoints/brain_mri_efficientnet.pth"):
    global _active_model
    unload_other_models('brain')
    device = get_device()
    if _active_model is None:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 4)
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"[Model Loaded] Successfully restored Brain MRI weights from {model_path}")
            except Exception as e:
                print(f"[Model Warning] Failed loading {model_path}: {e}")
        else:
            print(f"[Model Warning] Checkpoint {model_path} not found. Using pretrained EfficientNet-B0 weights.")

        model.to(device)
        model.eval()
        _active_model = model
    return _active_model, device

def predict_brain_mri(image_bytes, model_path="models_checkpoints/brain_mri_efficientnet.pth"):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    pil_img_resized = pil_img.resize((224, 224))
    
    is_valid, reason = validate_scan_domain(pil_img_resized, 'brain')
    if not is_valid:
        return {
            "label": "CANNOT BE DIAGNOSED",
            "confidence": 0.0,
            "class_probabilities": {},
            "gradcam_base64": None,
            "is_valid": False,
            "reason": reason
        }

    model, device = get_brain_model(model_path)
    rgb_img = np.float32(pil_img_resized) / 255.0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)

    target_layer = model.features[-1]
    if _USE_PYTORCH_GRAD_CAM:
        try:
            cam = GradCAM(model=model, target_layers=[target_layer])
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        except Exception:
            grayscale_cam = compute_native_gradcam(model, target_layer, input_tensor, pred_idx)
            visualization = overlay_heatmap_on_image(rgb_img, grayscale_cam)
    else:
        grayscale_cam = compute_native_gradcam(model, target_layer, input_tensor, pred_idx)
        visualization = overlay_heatmap_on_image(rgb_img, grayscale_cam)

    buffered = io.BytesIO()
    Image.fromarray(visualization).save(buffered, format="PNG")
    gradcam_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    prob_dict = {classes[i]: round(float(probs[i]) * 100, 2) for i in range(len(classes))}

    return {
        "label": label,
        "confidence": float(confidence),
        "class_probabilities": prob_dict,
        "gradcam_base64": gradcam_b64
    }

# ----------------------------------------------------
# 4. Bone Fracture X-Ray Model (ResNet-50)
# ----------------------------------------------------
def get_bone_model(model_path="models_checkpoints/bone_fracture_resnet50.pth"):
    global _active_model
    unload_other_models('bone')
    device = get_device()
    if _active_model is None:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"[Model Loaded] Successfully restored Bone Fracture weights from {model_path}")
            except Exception as e:
                print(f"[Model Warning] Failed loading {model_path}: {e}")
        else:
            print(f"[Model Warning] Checkpoint {model_path} not found. Using pretrained ResNet-50 weights.")

        model.to(device)
        model.eval()
        _active_model = model
    return _active_model, device

def predict_bone_fracture(image_bytes, model_path="models_checkpoints/bone_fracture_resnet50.pth"):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    pil_img_resized = pil_img.resize((224, 224))
    
    is_valid, reason = validate_scan_domain(pil_img_resized, 'bone')
    if not is_valid:
        return {
            "label": "CANNOT BE DIAGNOSED",
            "confidence": 0.0,
            "class_probabilities": {},
            "gradcam_base64": None,
            "is_valid": False,
            "reason": reason
        }

    model, device = get_bone_model(model_path)
    rgb_img = np.float32(pil_img_resized) / 255.0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    classes = ['fractured', 'not fractured']
    label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)

    target_layer = model.layer4[-1]
    if _USE_PYTORCH_GRAD_CAM:
        try:
            cam = GradCAM(model=model, target_layers=[target_layer])
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        except Exception:
            grayscale_cam = compute_native_gradcam(model, target_layer, input_tensor, pred_idx)
            visualization = overlay_heatmap_on_image(rgb_img, grayscale_cam)
    else:
        grayscale_cam = compute_native_gradcam(model, target_layer, input_tensor, pred_idx)
        visualization = overlay_heatmap_on_image(rgb_img, grayscale_cam)

    buffered = io.BytesIO()
    Image.fromarray(visualization).save(buffered, format="PNG")
    gradcam_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "label": label,
        "confidence": float(confidence),
        "gradcam_base64": gradcam_b64
    }
