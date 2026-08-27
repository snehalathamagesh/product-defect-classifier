# Metal Cast Defect Classifier & MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)

An end-to-end computer vision solution designed to detect and classify metal cast manufacturing defects (`defective` vs. `ok`). Built with PyTorch, FastAPI, Streamlit, and an MLOps architecture featuring data validation, modularized feature extraction, and dynamic distribution drift benchmarking.

---

##  Project Overview

This project provides a complete deep learning workflow:
* **Deep Learning Framework:** Compares a baseline Custom CNN against a pre-trained ResNet-18 Transfer Learning model.
* **REST API:** Serves real-time inference using FastAPI.
* **Web UI:** Provides an interactive operational dashboard built with Streamlit.
* **MLOps Core:** Features input image validation, isolated feature engineering, and automated environment drift simulation (dim lighting, camera lens blur, conveyor tilt).

---

## 📁 Repository Structure

```text
defect_classifier/
├── data/
│   └── raw/
│       ├── defective/          # Defective PCB images
│       └── ok/                 # Defect-free PCB images
├── training/
│   └── train.py                # ResNet-18 transfer learning training script
├── models/
│   └── resnet_model.pth        # Saved PyTorch model checkpoint
├── serving/
│   └── api.py                  # FastAPI server for inference
├── ui/
│   └── app.py                  # Streamlit visual dashboard
├── validation/
│   └── validate_images.py      # Input file sanity checks
├── features/
│   └── preprocess.py           # Feature transformer pipeline
├── src/
│   └── simulate_drift.py       # Operational drift benchmarking simulator
└── logs/
    └── drift_benchmark_results.csv # Evaluation telemetry output
```

## Quick Run Commands

Execute the entire end-to-end pipeline—from workspace setup to training, backend API serving, frontend UI launching, and MLOps drift simulation—by running the following commands:

```powershell
# 1. Clone & enter project folder
git clone [https://github.com/your-username/defect_classifier.git](https://github.com/your-username/defect_classifier.git)
cd defect_classifier

# 2. Setup isolated Python 3.13 virtual environment & activate
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install torch torchvision pandas numpy pillow fastapi uvicorn streamlit requests scikit-learn

# 4. Train the ResNet-18 Transfer Learning model
python training/train.py

# 5. Launch the REST API backend (Serves at [http://127.0.0.1:8000](http://127.0.0.1:8000))
python serving/api.py

# 6. Launch Streamlit Web UI (In a new terminal with venv active; opens at http://localhost:8501)
streamlit run ui/app.py

# 7. Run MLOps dynamic distribution drift simulation & export telemetry to logs/
python src/simulate_drift.py
