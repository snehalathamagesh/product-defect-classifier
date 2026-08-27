import os
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Import shared preprocessing from feature management layer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.preprocess import get_inference_transforms

IMAGE_TRANSFORMS = get_inference_transforms()

class DefectDataset(Dataset):
    def __init__(self, raw_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        categories = {'ok': 0, 'defective': 1}
        
        for category, label in categories.items():
            cat_dir = os.path.join(raw_dir, category)
            if not os.path.exists(cat_dir):
                continue
                
            for img_name in os.listdir(cat_dir):
                img_path = os.path.join(cat_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(data_dir="data/raw", batch_size=32, val_split=0.2):
    full_dataset = DefectDataset(raw_dir=data_dir, transform=IMAGE_TRANSFORMS)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader