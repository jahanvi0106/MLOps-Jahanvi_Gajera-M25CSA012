import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model
import warnings
warnings.filterwarnings('ignore')

def setup_model_with_lora(rank=4, alpha=8, dropout=0.1, train_head_only=False):
    """
    Setup ViT model with optional LoRA configuration.
    If rank is None -> full fine-tuning (no LoRA).
    """
    # Load pre-trained ViT-small model (pre-trained on ImageNet-21k)
    model = ViTForImageClassification.from_pretrained(
        'cs-giung/vit-small-patch16-imagenet21k-augreg',
        ignore_mismatched_sizes=True,
        num_labels=100
    )
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # If rank is None -> full fine-tuning (only classifier head trainable)
    if rank is None:
        for param in model.classifier.parameters():
            param.requires_grad = True
        print(f"Full fine-tuning - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model
    
    # Otherwise, apply LoRA
    if train_head_only:
        for param in model.classifier.parameters():
            param.requires_grad = True
        print(f"Trainable parameters (head only): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model
    
    # Configure LoRA for attention layers
    lora_config = LoraConfig(
        r=rank,                     # Rank of adaptation
        lora_alpha=alpha,           # Scaling parameter
        lora_dropout=dropout,       # Dropout rate
        target_modules=["query", "key", "value"],  # Inject LoRA into Q, K, V
        bias="none",
        modules_to_save=["classifier"]  # Keep classification head trainable
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def count_trainable_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params