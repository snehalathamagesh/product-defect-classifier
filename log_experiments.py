import os
import csv
import json
import torch
import torch.nn as nn
from torchvision import models
import sys

# Ensure data_loader can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from training.data_loader import get_data_loaders

# Architectures
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

def evaluate_model(model, val_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = round(running_loss / len(val_loader), 4)
    accuracy = round(100 * correct / total, 2)
    return avg_loss, accuracy

def log_run(model_name, epochs, lr, batch_size, train_loss, val_acc):
    LOG_DIR = "logs"
    CSV_FILE = os.path.join(LOG_DIR, "experiment_history.csv")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Epochs", "LearningRate", "BatchSize", "TrainLoss", "ValAccuracy"])
        writer.writerow([model_name, epochs, lr, batch_size, train_loss, val_acc])
    print(f"✅ Evaluated & Logged {model_name} | Val Loss: {train_loss} | Val Acc: {val_acc}%")

if __name__ == "__main__":
    _, val_loader = get_data_loaders("data/raw", batch_size=32)

    # 1. Evaluate Simple CNN
    cnn = SimpleDefectCNN()
    if os.path.exists("models/simple_cnn.pth"):
        cnn.load_state_dict(torch.load("models/simple_cnn.pth", weights_only=True))
        loss, acc = evaluate_model(cnn, val_loader)
        log_run("Custom_CNN", epochs=5, lr=0.0001, batch_size=32, train_loss=loss, val_acc=acc)

    # 2. Evaluate ResNet-18
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    if os.path.exists("models/resnet_model.pth"):
        resnet.load_state_dict(torch.load("models/resnet_model.pth", weights_only=True))
        loss, acc = evaluate_model(resnet, val_loader)
        log_run("ResNet18_Transfer", epochs=5, lr=0.0001, batch_size=32, train_loss=loss, val_acc=acc)