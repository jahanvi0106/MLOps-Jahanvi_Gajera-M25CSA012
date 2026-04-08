import optuna
import torch
import wandb
from que1.data_loader import get_cifar100_loaders
from que1.model import setup_model_with_lora
from que1.train import Trainer

def objective(trial):
    """Optuna objective function for LoRA hyperparameters"""
    
    # Suggest hyperparameters
    rank = trial.suggest_categorical('rank', [2, 4, 8])
    alpha = trial.suggest_categorical('alpha', [2, 4, 8])
    dropout = 0.1  # Fixed as per assignment
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _ = get_cifar100_loaders(batch_size=64)
    
    # Create model with LoRA
    model = setup_model_with_lora(rank=rank, alpha=alpha, dropout=dropout)
    
    # Train
    trainer = Trainer(
        model, train_loader, val_loader, device,
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    
    # Train for fewer epochs during optimization
    history, _ = trainer.train(epochs=5, log_wandb=False)
    
    # Return validation accuracy (to maximize)
    return max(history['val_acc'])

def run_optuna_optimization(n_trials=30):
    """Run Optuna hyperparameter optimization"""
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='lora_optimization',
        storage='sqlite:///optuna_study.db',
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials)
    
    # Print best results
    print("Best trial:")
    print(f"  Value (Val Acc): {study.best_value:.2f}%")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    return study