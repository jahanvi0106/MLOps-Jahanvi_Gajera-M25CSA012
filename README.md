
---

## Model Used

Model: `prajjwal1/bert-tiny`

### Why DistilBERT?

- Lightweight and faster than BERT  
- Lower computational cost  
- Good baseline for text classification  
- Fully supported by Hugging Face Trainer API  
- Suitable for Docker deployment  

---

## 🏋️ Training Configuration

- Framework: Hugging Face Transformers  
- Optimizer: AdamW  
- Loss Function: CrossEntropyLoss  
- Evaluation Strategy: Per Epoch  
- Metrics: Accuracy, Weighted F1-score  

---

## 📊 Evaluation Results

### 🔹 Local Evaluation

| Metric | Value |
|--------|--------|
| Eval Loss | 3.2344 |
| Accuracy | 0.1444 |
| Weighted F1 | 0.1441 |

---

### Evaluation from Hugging Face Hub

| Metric | Value |
|--------|--------|
| Eval Loss | 2.8062 |
| Accuracy | 0.1288 |
| Weighted F1 | 0.1175 |

---

## Performance Comparison

| Metric | Local Model | HF Hub Model |
|--------|-------------|--------------|
| Eval Loss | 3.2344 | 2.8062 |
| Accuracy | 0.1444 | 0.1288 |
| Weighted F1 | 0.1441 | 0.1175 |

Minor differences may occur due to environment configuration or random seed variations.

---

## Docker Usage

### Build Training Image

```bash
docker build -f Dockerfile.train -t hf-train .
docker run  hf-train
