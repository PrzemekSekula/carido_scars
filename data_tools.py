"""
This module contains the custom dataset class and the data module class 
for loading and preparing image data.
Autor: Przemek Sekula
Created: 2025-10-04
Last modified: 2025-10-04
"""

# %%
from collections import Counter
import torch
from torch.utils.data import Dataset
from PIL import Image
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    """
    Custom dataset class for loading images and their labels.
    """
    def __init__(self, file_list, class_to_idx, transform=None):
        self.file_list = file_list
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, class_name = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[class_name]
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading and preparing image data.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = self.config["data_dir"]
        self.image_size = self.config["image_size"]
        self.batch_size = self.config["batch_size"]
        self.num_workers = self.config["num_workers"]
        self.class_weights = None # Initialize the attribute
        
        self.train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.val_test_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        filepaths = list(Path(self.data_dir).rglob('*.jpeg'))
        class_names = [path.parent.parent.name for path in filepaths]
        all_files = list(zip(filepaths, class_names))
        
        self.class_to_idx = {name: i for i, name in enumerate(sorted(set(class_names)))}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
        print(f"Found classes: {self.class_to_idx}")
        
        train_val_files, self.test_files = train_test_split(
            all_files, test_size=self.config["test_split_size"], stratify=[f[1] for f in all_files], random_state=42
        )
        val_size_adjusted = self.config["val_split_size"] / (1 - self.config["test_split_size"])
        self.train_files, self.val_files = train_test_split(
            train_val_files,
            test_size=val_size_adjusted,
            stratify=[f[1] for f in train_val_files],
            random_state=42
        )
        
        if self.config.get("use_class_weights", False):
            train_labels = [label for _, label in self.train_files]
            class_counts = Counter(train_labels)
            
            # Sort class names to ensure weight order matches class index order
            sorted_class_names = sorted(class_counts.keys())
            
            # Calculate weights using the inverse frequency formula
            total_samples = len(train_labels)
            num_classes = len(sorted_class_names)
            weights = [total_samples / (num_classes * class_counts[cls]) for cls in sorted_class_names]
            
            self.class_weights = torch.FloatTensor(weights)
            
            print(f"✅ Using class weights: {self.class_weights}")
            for i, class_name in enumerate(sorted_class_names):
                print(f"   - Weight for class '{class_name}' (index {i}): {self.class_weights[i]:.2f}")


        print(f"Total images: {len(all_files)}")
        print(f"Training images: {len(self.train_files)}")
        print(f"Validation images: {len(self.val_files)}")
        print(f"Testing images: {len(self.test_files)}")
        
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomImageDataset(self.train_files, self.class_to_idx, transform=self.train_transform)
            self.val_dataset = CustomImageDataset(self.val_files, self.class_to_idx, transform=self.val_test_transform)
        if stage == 'test' or stage is None:
            self.test_dataset = CustomImageDataset(self.test_files, self.class_to_idx, transform=self.val_test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
