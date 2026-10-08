---
title: XAI MedVision 🩺 Explainable AI Diagnostics
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multi-modal Explainable AI Medical Vision & Clinical NLP Platform
---

# XAI-MedVision 🩺 Dynamic Explainable AI Medical Diagnostics Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**XAI-MedVision** is a multi-modal, Explainable Artificial Intelligence (XAI) platform designed for clinical diagnostic assistance. It integrates Deep Learning computer vision models with Grad-CAM visual explanations alongside Natural Language Processing (NLP) models for automated medical report summarization.

---

## 👥 Authors & Contributors

- **Jammula Rithik Reddy** ([@crazynani1432](https://github.com/crazynani1432))
- **Pilli Sai Medhas** ([@pillisaimedhas-dev](https://github.com/pillisaimedhas-dev))

---

## 🌟 Key Features

### 👁️ Multi-Modal Computer Vision Suite (with Grad-CAM Heatmaps)
1. **Chest X-Ray Diagnostic Model (DenseNet-121)**: Detects Pneumonia vs. Normal pulmonary scans with heatmaps pointing to lung consolidations.
2. **Blood Cell CBC Classifier (MobileNet-V2)**: Classifies blood cell types (Eosinophil, Lymphocyte, Monocyte, Neutrophil) with cellular localization overlays.
3. **Brain MRI Classifier (EfficientNet-B0)**: Classifies MRI scans into Glioma, Meningioma, Pituitary tumor, or No Tumor with tumor bounding overlays.
4. **Bone Fracture Classifier (ResNet-50)**: Identifies musculoskeletal bone fractures with fracture line feature highlighting.

### 📝 Clinical NLP & Report Summarization
- **FLAN-T5 LoRA Fine-Tuned Model**: fine-tuned on clinical report summaries (MIMIC dataset format) for automatic medical text abstraction and key clinical recommendations.

---

## 🏗️ Project Architecture

```
XAI-MedVision project/
├── app/
│   ├── main.py                    # FastAPI Web Server & API Endpoints
│   ├── services/
│   │   ├── vision_models.py       # Computer Vision Models & Grad-CAM pipeline
│   │   └── nlp_service.py          # FLAN-T5 LoRA Summarization pipeline
│   └── static/                    # Frontend Web Interface (HTML5 / Vanilla CSS / JS)
├── data/                          # Dataset files and preprocessing scripts
│   └── mimic_cleaned_summaries.csv
├── models_checkpoints/            # Weights directory (.pth, .safetensors)
├── prepare_mimic.py               # MIMIC NLP dataset preparer
├── train_blood_cells.py           # Training script for Blood Cell MobileNet
├── train_bone_fracture.py         # Training script for Bone Fracture ResNet50
├── train_brain_mri.py             # Training script for Brain MRI EfficientNet
├── train_chest_xray.py            # Training script for Chest X-Ray DenseNet
├── train_mimic_nlp.py             # Fine-tuning script for FLAN-T5 LoRA model
└── README.md                      # Project Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ and Git installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/crazynani1432/XAI-MedVision.git
cd XAI-MedVision
```

### 3. Create Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Running the Web Application
Launch the FastAPI application server:
```bash
uvicorn app.main:app --reload --port 8000
```
Open your browser and navigate to:
- **Web UI**: `http://localhost:8000/`
- **Swagger API Documentation**: `http://localhost:8000/docs`

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
