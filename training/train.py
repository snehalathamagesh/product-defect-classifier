import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_loader import get_data_loaders

# 1. Custom CNN (Trained from Scratch)
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

def run_epoch(model, loader, criterion, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
        
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            if optimizer:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if optimizer:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss, 100 * correct / total

def train_and_save(model, name, save_path, train_loader, val_loader, epochs=5, lr=0.0001):
    print(f"--- Training {name} ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        _, val_acc = run_epoch(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved {name} weights to '{save_path}'\n")

if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders("data/raw", batch_size=32)

    # Train Model 1: Custom CNN
    cnn = SimpleDefectCNN()
    train_and_save(cnn, "Custom CNN", "models/simple_cnn.pth", train_loader, val_loader)

    # Train Model 2: Transfer Learning (ResNet-18)
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    train_and_save(resnet, "ResNet-18 (Transfer Learning)", "models/resnet_model.pth", train_loader, val_loader)