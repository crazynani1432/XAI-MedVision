# 🚀 XAI-MedVision Deployment Guide

This guide provides step-by-step instructions for deploying **XAI-MedVision**, a multi-modal Explainable AI Medical Diagnostics Platform, across local, containerized, and cloud environments.

---

## 📋 Prerequisites

- **Python**: 3.10 or 3.11 (Python 3.11 recommended for PyTorch CPU compatibility)
- **Docker** (Optional, for containerized deployment)
- **Git**

---

## 💻 1. Local Deployment (Windows / Linux / macOS)

### Quick Start (Windows)
Double-click `start_app.bat` or run in PowerShell / Command Prompt:
```cmd
.\start_app.bat
```

### Manual Setup
1. **Create and Activate Virtual Environment**:
   ```bash
   # Windows:
   py -3.11 -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch FastAPI Web Application**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the Application**:
   - **Web UI Dashboard**: `http://localhost:8000/`
   - **Interactive API Documentation (Swagger)**: `http://localhost:8000/docs`
   - **Health Check Endpoint**: `http://localhost:8000/api/health`

---

## 🐳 2. Containerized Deployment (Docker & Docker Compose)

### Using Docker Compose (Recommended)
```bash
# Build and start container in background
docker-compose up -d --build

# View container logs
docker-compose logs -f

# Stop container
docker-compose down
```
Access at: `http://localhost:7860/`

### Using Standalone Docker CLI
```bash
# Build image
docker build -t xai-medvision .

# Run container
docker run -d -p 7860:7860 --name xai-medvision-app xai-medvision
```

---

## 🤗 3. Deploying to Hugging Face Spaces (Free Cloud Hosting)

Hugging Face Spaces provides free, continuous cloud hosting for Docker applications.

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Select **Docker** as the Space SDK.
3. Choose **Blank** template and set visibility to Public or Private.
4. Push the project repository to your Hugging Face Space Git repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/XAI-MedVision
   git push hf main
   ```
5. Hugging Face will automatically read `Dockerfile`, build the container, and host the web app at `https://huggingface.co/spaces/YOUR_USERNAME/XAI-MedVision` on port 7860!

---

## ☁️ 4. Deploying to Render / Railway / Cloud VMs

### Render Deployment
This project includes a `render.yaml` specification for zero-config Render deployment:
1. Connect your GitHub repository to [Render.com](https://render.com).
2. Create a new **Web Service** selecting `render.yaml`.
3. Render will execute:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

---

## 🔌 5. API Endpoints Reference

| Endpoint | Method | Input | Description |
| :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | None | Returns system health & pre-warmed model status |
| `/api/predict/brain-mri` | `POST` | Image file | Brain Tumor MRI classification + Grad-CAM heatmap |
| `/api/predict/bone-fracture` | `POST` | Image file | Musculoskeletal fracture detection + Grad-CAM heatmap |
| `/api/predict/chest-xray` | `POST` | Image file | Pneumonia detection + Grad-CAM heatmap |
| `/api/predict/blood-cell` | `POST` | Image file | Complete Blood Count differential + Grad-CAM heatmap |
| `/api/summarize/clinical-note` | `POST` | JSON `{ "text": "..." }` | FLAN-T5 LoRA clinical discharge note summarizer |

---

## 🛡️ Production Best Practices

1. **Model Weight Pre-warming**: `app/main.py` pre-warms PyTorch weights into memory during application startup to guarantee $<150\text{ ms}$ response latencies on first request.
2. **CORS Configuration**: CORS middleware is enabled to support external frontend applications or clinical EHR integrations.
