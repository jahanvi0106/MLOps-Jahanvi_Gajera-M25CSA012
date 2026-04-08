import torch
import wandb
from q1.data_loader import get_cifar100_loaders
from q1.model import setup_model_with_lora, count_trainable_parameters
from q1.train import Trainer
from q1.evaluate import evaluate_test

def run_experiment(rank, alpha, dropout, experiment_name, log_wandb=True):
    """Run a single LoRA experiment"""
    
    # Initialize WandB
    if log_wandb:
        wandb.init(
            project="dlops-assignment5",
            name=experiment_name,
            config={
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
                "model": "ViT-S",
                "dataset": "CIFAR-100"
            }
        )
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_cifar100_loaders(batch_size=16)
    
    # Create model
    model = setup_model_with_lora(rank=rank, alpha=alpha, dropout=dropout)
    total_params, trainable_params = count_trainable_parameters(model)
    
    print(f"Experiment: {experiment_name}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, device)
    history, class_acc = trainer.train(epochs=10, log_wandb=log_wandb)
    
    # Test
    test_acc, _, _ = evaluate_test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Log test results
    if log_wandb:
        wandb.log({"test_accuracy": test_acc})
        wandb.finish()
    
    return {
        'experiment': experiment_name,
        'rank': rank,
        'alpha': alpha,
        'dropout': dropout,
        'trainable_params': trainable_params,
        'history': history,
        'test_acc': test_acc,
        'class_acc': class_acc
    }

def run_all_experiments():
    """Run all combinations of rank and alpha"""
    
    experiments = []
    
    # Full fine-tuning (without LoRA)
    # print("\n" + "="*50)
    # print("Running Full Fine-tuning (without LoRA)")
    # print("="*50)
    # result = run_experiment(rank=None, alpha=None, dropout=None, 
    #                        experiment_name="full_finetune")
    # experiments.append(result)
    
    # LoRA experiments with different ranks and alphas
    ranks = [2, 4, 8]
    alphas = [2, 4, 8]
    
    for rank in ranks:
        for alpha in alphas:
            print("\n" + "="*50)
            print(f"Running LoRA Experiment: r={rank}, alpha={alpha}")
            print("="*50)
            result = run_experiment(
                rank=rank, alpha=alpha, dropout=0.1,
                experiment_name=f"lora_r{rank}_a{alpha}"
            )
            experiments.append(result)
    
    return experiments

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run all experiments
    results = run_all_experiments()
    
    # Print summary table
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Experiment':<20} {'Rank':<6} {'Alpha':<6} {'Test Acc':<10} {'Trainable Params':<15}")
    print("-"*60)
    for r in results:
        exp_name = r['experiment']
        rank = r['rank'] if r['rank'] else 'N/A'
        alpha = r['alpha'] if r['alpha'] else 'N/A'
        test_acc = f"{r['test_acc']:.2f}%"
        params = f"{r['trainable_params']:,}"
        print(f"{exp_name:<20} {str(rank):<6} {str(alpha):<6} {test_acc:<10} {params:<15}")