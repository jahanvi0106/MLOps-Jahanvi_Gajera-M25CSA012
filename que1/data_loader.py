import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_cifar100_loaders(batch_size=16, num_workers=0, val_split=0.1):
    """Load CIFAR-100 dataset with train/val split"""
    
    # Transformations for ViT (resize to 224x224 as ViT expects)
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load full training dataset
    full_train = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [train_size, val_size]
    )
    
    # Apply validation transform
    val_dataset.dataset.transform = val_transform
    
    # Load test dataset
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader