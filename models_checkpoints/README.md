# Model Checkpoints Directory

This directory is intended to store trained model checkpoints for XAI-MedVision:

- `blood_cell_mobilenet.pth`: MobileNetV2 trained on Blood Cell Images.
- `bone_fracture_resnet50.pth`: ResNet50 model for Bone Fracture detection.
- `brain_mri_efficientnet.pth`: EfficientNet model for Brain Tumor classification on MRI.
- `chest_xray_densenet.pth`: DenseNet121 model for Chest X-Ray diagnostic classification.
- `mimic_flan_t5_lora/`: FLAN-T5 LoRA adapter weights for MIMIC medical report summarization.

> Note: Model weights (`.pth`, `.safetensors`, `.bin`) are excluded from version control due to file size limits. Place downloaded weights in this directory before launching the app.
