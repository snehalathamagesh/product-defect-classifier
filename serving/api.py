import io
import os
import sys
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from PIL import Image, UnidentifiedImageError
from torchvision import models

# Ensure features can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.preprocess import get_inference_transforms

app = FastAPI(
    title="Defect Classifier API",
    description="Production REST API for real-time manufacturing defect detection",
    version="1.0.0"
)

IMAGE_TRANSFORMS = get_inference_transforms()
CLASS_NAMES = ["ok", "defective"]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

# Model Definitions
class SimpleDefectCNN(nn.Module):
    def __init__(self):
        super(SimpleDefectCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Load Models Safely
cnn_model = SimpleDefectCNN()
resnet_model = models.resnet18(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)

try:
    if os.path.exists("models/simple_cnn.pth"):
        cnn_model.load_state_dict(torch.load("models/simple_cnn.pth", map_location="cpu", weights_only=True))
        cnn_model.eval()
    if os.path.exists("models/resnet_model.pth"):
        resnet_model.load_state_dict(torch.load("models/resnet_model.pth", map_location="cpu", weights_only=True))
        resnet_model.eval()
except Exception as e:
    print(f"Warning: Model weight loading encountered an issue: {e}")

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Defect Classifier REST API"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Query("resnet", description="Select model: 'cnn' or 'resnet'")
):
    # 1. Edge-Case: Check file extension
    filename = file.filename or ""
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 2. Edge-Case: Read file content & validate non-empty payload
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    # 3. Edge-Case: Handle corrupted or unreadable images
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Corrupted or invalid image file. Could not decode pixel data."
        )

    # 4. Select Model
    if model_type.lower() not in ["cnn", "resnet"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model_type requested. Choose 'cnn' or 'resnet'."
        )

    active_model = cnn_model if model_type.lower() == "cnn" else resnet_model

    # 5. Execute Skew-Free Inference
    try:
        tensor = IMAGE_TRANSFORMS(image).unsqueeze(0)
        with torch.no_grad():
            outputs = active_model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failed: {str(e)}"
        )

    return {
        "filename": file.filename,
        "model_used": "Custom CNN" if model_type.lower() == "cnn" else "ResNet-18",
        "prediction": CLASS_NAMES[predicted_idx.item()],
        "confidence_percentage": round(confidence.item() * 100, 2)
    }