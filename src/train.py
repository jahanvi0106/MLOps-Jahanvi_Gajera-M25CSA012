# train.py

import os
import torch
import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


from sklearn.metrics import accuracy_score, f1_score

from data import prepare_datasets


# =====================================
# Disable WandB (if not required)
# =====================================
os.environ["WANDB_DISABLED"] = "true"


# =====================================
# Configuration
# =====================================
MODEL_NAME = "prajjwal1/bert-tiny"
# MODEL_NAME = "Jahanvi16/tinybert_model"

OUTPUT_DIR = "./logs_epoch_local"
MODEL_SAVE_PATH = "model_epoch_local"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"






# =====================================
# Compute Metrics (Logging Metrics)
# =====================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1,
    }


# =====================================
# Main Training Function
# =====================================
def main():

    # ---------------------------------
    # 1️⃣ Prepare Dataset
    # ---------------------------------
    train_dataset, eval_dataset, label2id, id2label = prepare_datasets()

    # ---------------------------------
    # 2️⃣ Load Pretrained Model
    # ---------------------------------
    from transformers import BertConfig, BertForSequenceClassification
    from transformers import AutoTokenizer
    config = BertConfig.from_pretrained(
        "prajjwal1/bert-tiny",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True
    )


    model.to(DEVICE)

    # ---------------------------------
    # 3️⃣ Configure Training Arguments
    # ---------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=16,
        learning_rate=5e-4,
        warmup_steps=100,
        weight_decay=0.01,

        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        report_to=[],   # Disable wandb
        save_total_limit=2,
    )

    # ---------------------------------
    # 4️⃣ Initialize Trainer API
    # ---------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # ---------------------------------
    # 5️⃣ Train Model Using Trainer API
    # ---------------------------------
    print("Starting training...")
    train_result = trainer.train()

    # ---------------------------------
    # 6️⃣ Log Training Metrics
    # ---------------------------------
    trainer.save_model(MODEL_SAVE_PATH)

    # ---------------------------------
    # 7️⃣ Save as .pth (PyTorch format)
    # ---------------------------------
    torch.save(model.state_dict(), "model.pth")
    print(f"\nModel saved as {MODEL_SAVE_PATH}")
   
    model.config.save_pretrained("./")


    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\nTraining Complete.")
    print("Training Metrics:")
    print(metrics)


# =====================================
# Run Script
# =====================================
if __name__ == "__main__":
    main()
