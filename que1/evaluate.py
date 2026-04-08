import torch
import numpy as np
from tqdm import tqdm

def evaluate_test(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.logits.max(1)
            
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    return accuracy, all_preds, all_targets

def compute_gradient_stats(model):
    """Compute gradient statistics for LoRA weights"""
    grad_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    return {
        'mean_grad_norm': np.mean(grad_norms),
        'std_grad_norm': np.std(grad_norms),
        'max_grad_norm': np.max(grad_norms)
    }