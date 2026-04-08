import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, AutoConfig
from peft import get_peft_model, LoraConfig
import wandb
import optuna
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast, GradScaler 
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
NUM_CLASSES = 100
BATCH_SIZE = 128
EPOCHS = 10 # Adjust as needed for your compute budget
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using a standard ViT-Small model available on HuggingFace
MODEL_ID = "cs-giung/vit-small-patch16-imagenet21k-augreg"

# ==========================================
# DATA LOADING
# ==========================================
def get_dataloaders():
    # ViT models typically expect 224x224 input images
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    # Only download if not already present
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return trainloader, testloader, testset.classes

# ==========================================
# TRAINING & EVALUATION FUNCTIONS
# ==========================================
def plot_class_histogram(all_preds, all_labels, classes, step, run_name):
    correct_per_class = np.zeros(NUM_CLASSES)
    total_per_class = np.zeros(NUM_CLASSES)
    
    for p, l in zip(all_preds, all_labels):
        total_per_class[l] += 1
        if p == l:
            correct_per_class[l] += 1
            
    acc_per_class = (correct_per_class / total_per_class) * 100
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(classes, acc_per_class)
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Class-wise Test Accuracy - {run_name}')
    plt.tight_layout()
    
    wandb.log({f"Class-wise_Histogram_{run_name}": wandb.Image(fig)}, step=step)
    plt.close(fig)


def train_epoch(model, dataloader, criterion, optimizer, epoch, log_lora_grads=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    lora_grads_epoch = []
    scaler = GradScaler()
    
    for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch} Training"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        with autocast():  # ← wrap forward pass
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        
        # Log Gradient updates for LoRA weights (per-batch) and accumulate for epoch
        if log_lora_grads:
            lora_grads = []
            for name, param in model.named_parameters():
                if "lora" in name and param.grad is not None:
                    lora_grads.append(param.grad.abs().mean().item())
            if lora_grads:
                batch_mean = np.mean(lora_grads)
                wandb.log({"LoRA_Gradient_Update_Mean_batch": batch_mean})
                lora_grads_epoch.append(batch_mean)
        
        scaler.step(optimizer)   # ← replace optimizer.step()
        scaler.update()   
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_acc = 100. * correct / total
    train_loss = running_loss / len(dataloader)
    lora_grad_mean = None
    if lora_grads_epoch:
        lora_grad_mean = float(np.mean(lora_grads_epoch))
    return train_loss, train_acc, lora_grad_mean

def eval_model(model, dataloader, criterion, classes, epoch, run_name, plot_hist=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch} Eval"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if plot_hist:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
    val_acc = 100. * correct / total
    val_loss = running_loss / len(dataloader)
    
    if plot_hist:
        plot_class_histogram(all_preds, all_labels, classes, epoch, run_name)
        
    return val_loss, val_acc

# ==========================================
# BASELINE (NO LORA)
# ==========================================
def run_baseline(trainloader, testloader, classes):
    wandb.init(project="DLOps-Ass5-Q1", name="Baseline_No_LoRA")
    
    # configure model dropout via config
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1
    config.num_labels = NUM_CLASSES

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    # Freeze all layers EXCEPT the classification head
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    # collect per-epoch metrics
    epoch_rows = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    lora_grad_means = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, lora_grad_mean = train_epoch(model, trainloader, criterion, optimizer, epoch, log_lora_grads=False)
        plot_hist = (epoch == EPOCHS) # Plot histogram on last epoch
        val_loss, val_acc = eval_model(model, testloader, criterion, classes, epoch, "Baseline", plot_hist)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        lora_grad_means.append(lora_grad_mean)
        epoch_rows.append([epoch, train_loss, train_acc, val_loss, val_acc, lora_grad_mean])

         # Print epoch metrics to console
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        wandb.log({
            "epoch": epoch,
            "Train Loss": train_loss,
            "Train Acc": train_acc,
            "Val Loss": val_loss,
            "Val Acc": val_acc
        })
    
    # Log a table of epoch metrics
    table = wandb.Table(columns=["epoch","train_loss","train_acc","val_loss","val_acc","lora_grad_mean"], data=epoch_rows)
    wandb.log({"Baseline_Metrics_Table": table})

    # Create and log loss/accuracy plots
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    axs[0].plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    axs[0].plot(range(1, EPOCHS+1), val_losses, label='Val Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(range(1, EPOCHS+1), train_accs, label='Train Acc')
    axs[1].plot(range(1, EPOCHS+1), val_accs, label='Val Acc')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    wandb.log({"Baseline_Loss_Acc_Plots": wandb.Image(fig)})
    plt.close(fig)
    
    wandb.finish()

# ==========================================
# OPTUNA STUDY FOR LORA
# ==========================================
def objective(trial, trainloader, testloader, classes):
    # Search Space based on assignment instructions
    r = trial.suggest_categorical("r", [2, 4, 8])
    alpha = trial.suggest_categorical("alpha", [2, 4, 8])
    
    run_name = f"LoRA_Rank{r}_Alpha{alpha}"
    wandb.init(project="DLOps-Ass5-Q1", name=run_name, reinit=True)
    
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_labels = NUM_CLASSES

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID, 
        config=config,
        ignore_mismatched_sizes=True
    )
    
    # Inject LoRA into Q, K, V and keep classifier trainable
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q", "k", "v", "query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"] 
    )

    model = get_peft_model(model, lora_config)
    model = model.to(DEVICE)

    # Ensure classification head parameters are trainable
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    best_val_acc = 0.0

    # collect per-epoch metrics
    epoch_rows = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    lora_grad_means = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, lora_grad_mean = train_epoch(model, trainloader, criterion, optimizer, epoch, log_lora_grads=True)
        plot_hist = (epoch == EPOCHS)
        val_loss, val_acc = eval_model(model, testloader, criterion, classes, epoch, run_name, plot_hist)
        
        # Calculate trainable params only once per epoch
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        lora_grad_means.append(lora_grad_mean)
        epoch_rows.append([epoch, train_loss, train_acc, val_loss, val_acc, lora_grad_mean])
        
        # Print epoch metrics to console
        print(f"\n{run_name} - Epoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Trainable Params: {trainable_params:,} / {total_params:,}")
        if lora_grad_mean:
            print(f"  LoRA Grad Mean: {lora_grad_mean:.6f}")

        wandb.log({
            "epoch": epoch,
            "Train Loss": train_loss,
            "Train Acc": train_acc,
            "Val Loss": val_loss,
            "Val Acc": val_acc,
            "Rank": r,
            "Alpha": alpha,
            "Trainable_Params": trainable_params,
            "Total_Params": total_params,
            "LoRA_Grad_Epoch_Mean": lora_grad_mean
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the best model
            model.save_pretrained(f"./best_lora_model")
            
    # Log a table of epoch metrics
    table = wandb.Table(columns=["epoch","train_loss","train_acc","val_loss","val_acc","lora_grad_mean"], data=epoch_rows)
    wandb.log({f"{run_name}_Metrics_Table": table})

    # Create and log loss/accuracy plots
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    axs[0].plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    axs[0].plot(range(1, EPOCHS+1), val_losses, label='Val Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(range(1, EPOCHS+1), train_accs, label='Train Acc')
    axs[1].plot(range(1, EPOCHS+1), val_accs, label='Val Acc')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    wandb.log({f"{run_name}_Loss_Acc_Plots": wandb.Image(fig)})
    plt.close(fig)

    # Also log LoRA gradient mean curve
    if any(x is not None for x in lora_grad_means):
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(range(1, EPOCHS+1), [0 if x is None else x for x in lora_grad_means], marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('LoRA Gradient Mean')
        ax2.set_title(f'LoRA Gradient Mean per Epoch - {run_name}')
        wandb.log({f"{run_name}_LoRA_Grad_Plot": wandb.Image(fig2)})
        plt.close(fig2)

    wandb.finish()
    return best_val_acc

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure you are logged into wandb in your terminal: `wandb login`
    trainloader, testloader, classes = get_dataloaders()
    
    print("--- Running Baseline (No LoRA) ---")
    run_baseline(trainloader, testloader, classes)
    
    print("--- Running LoRA Optuna Hyperparameter Search ---")
    # Using GridSampler to strictly test all combinations of [2,4,8]
    search_space = {
        "r": [2, 4, 8],
        "alpha": [2, 4, 8]
    }
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction="maximize")
    
    # Pass arguments to objective function
    func = lambda trial: objective(trial, trainloader, testloader, classes)
    study.optimize(func, n_trials=9) # 3x3 = 9 combinations
    
    from huggingface_hub import HfApi

    HF_TOKEN    = "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"  # ← paste your token here
    HF_USERNAME = "Jahanvi16"                        # ← paste your HF username here
    REPO_NAME   = "Assign5-ViT-LoRA-CIFAR100"

    best_model_path = "./best_lora_model"

    print("Pushing best model to HuggingFace Hub...")
    api = HfApi(token=HF_TOKEN)

    try:
        # Create the repo if it doesn't exist yet
        api.create_repo(
            repo_id=f"{HF_USERNAME}/{REPO_NAME}",
            repo_type="model",
            exist_ok=True,          # no error if it already exists
        )

        api.upload_folder(
            folder_path=best_model_path,
            repo_id=f"{HF_USERNAME}/{REPO_NAME}",
            repo_type="model",
            token=HF_TOKEN,
            commit_message=(
                f"Best LoRA model — "
                f"Rank: {study.best_params['r']}, "
                f"Alpha: {study.best_params['alpha']}, "
                f"Acc: {study.best_value:.2f}%"
            ),
        )
        print(f"✓ Successfully pushed to: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")

    except Exception as e:
        print(f"✗ HuggingFace push failed: {e}")


    print("\n==============================================")
    print("Optuna Study Finished!")
    print(f"Best LoRA Configuration: {study.best_params}")
    print(f"Best Validation Accuracy: {study.best_value}%")
    print("Best weights are saved in the './best_lora_model' directory.")
    print("You can now push this directory to your GitHub and HuggingFace!")
    print("==============================================\n")
