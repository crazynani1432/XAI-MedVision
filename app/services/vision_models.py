import io
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

_chest_model = None
_blood_model = None
_brain_model = None
_bone_model = None
_device = None

def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

# ----------------------------------------------------
# 1. Chest X-Ray Pneumonia Model
# ----------------------------------------------------
def get_chest_model(model_path="models_checkpoints/chest_xray_densenet.pth"):
    global _chest_model
    device = get_device()
    if _chest_model is None:
        model = models.densenet121(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier.in_features, 2)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        _chest_model = model
    return _chest_model, device

def predict_chest_xray(image_bytes, model_path="models_checkpoints/chest_xray_densenet.pth"):
    model, device = get_chest_model(model_path)

    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    pil_img_resized = pil_img.resize((224, 224))
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

    target_layers = [model.features.denseblock4]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

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
    global _blood_model
    device = get_device()
    if _blood_model is None:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        _blood_model = model
    return _blood_model, device

def predict_blood_cell(image_bytes, model_path="models_checkpoints/blood_cell_mobilenet.pth"):
    model, device = get_blood_model(model_path)

    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    pil_img_resized = pil_img.resize((224, 224))
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

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

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
    global _brain_model
    device = get_device()
    if _brain_model is None:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        _brain_model = model
    return _brain_model, device

def predict_brain_mri(image_bytes, model_path="models_checkpoints/brain_mri_efficientnet.pth"):
    model, device = get_brain_model(model_path)

    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    pil_img_resized = pil_img.resize((224, 224))
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

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

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
    global _bone_model
    device = get_device()
    if _bone_model is None:
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        _bone_model = model
    return _bone_model, device

def predict_bone_fracture(image_bytes, model_path="models_checkpoints/bone_fracture_resnet50.pth"):
    model, device = get_bone_model(model_path)

    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    pil_img_resized = pil_img.resize((224, 224))
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

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    buffered = io.BytesIO()
    Image.fromarray(visualization).save(buffered, format="PNG")
    gradcam_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "label": label,
        "confidence": float(confidence),
        "gradcam_base64": gradcam_b64
    }
