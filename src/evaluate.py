# evaluate.py

import os
import json
import torch
import numpy as np

from transformers import AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data import prepare_datasets


# =====================================
# Configuration
# =====================================

# MODEL_PATH_h = os.environ.get("HF_MODEL_NAME", "prajjwal1/bert-tiny")
MODEL_PATH = os.environ.get("HF_MODEL_NAME", "Jahanvi16/tinybert_model")
RESULTS_DIR = "results_local"
RESULTS_FILE = os.path.join(RESULTS_DIR, "evaluation_results.json")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================
# Compute Metrics Function
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
# Main Evaluation
# =====================================
def main():

    # ---------------------------------
    # 1️⃣ Load Dataset (Test Split)
    # ---------------------------------
    _, test_dataset, label2id, id2label = prepare_datasets()

    # ---------------------------------
    # 2️⃣ Load Trained Model
    # ---------------------------------

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    MODEL_PATH = os.environ.get("HF_MODEL_NAME", "Jahanvi16/tinybert_model")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully!")

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    # ---------------------------------
    # 3️⃣ Run Evaluation
    # ---------------------------------
    print("Running evaluation...")
    eval_results = trainer.evaluate(test_dataset)

    print("\nEvaluation Metrics:")
    print(eval_results)

    # ---------------------------------
    # 4️⃣ Generate Predictions
    # ---------------------------------
    predictions_output = trainer.predict(test_dataset)

    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    preds = np.argmax(logits, axis=-1)

    # Compute detailed metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    report = classification_report(
        labels,
        preds,
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        output_dict=True
    )

    # ---------------------------------
    # 5️⃣ Save Evaluation Results
    # ---------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    final_results = {
        "eval_loss": eval_results.get("eval_loss"),
        "accuracy": accuracy,
        "f1_weighted": f1,
        "classification_report": report,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nEvaluation results saved to {RESULTS_FILE}")


# =====================================
# Run Script
# =====================================
if __name__ == "__main__":
    main()
