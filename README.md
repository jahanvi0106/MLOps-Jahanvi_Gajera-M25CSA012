# English → Hindi Transformer

**Model:** Custom Seq2Seq Transformer (PyTorch)  
**Task:** Machine Translation — English to Hindi  
**HuggingFace:** [Jahanvi16/en-to-hi-transformer](https://huggingface.co/Jahanvi16/en-to-hi-transformer)

---

## Quick Start

```python
from huggingface_hub import snapshot_download
from inference import load_model, translate

path = snapshot_download("Jahanvi16/en-to-hi-transformer")
model, en_vocab, hi_vocab = load_model(path)

print(translate(model, "How are you?", en_vocab, hi_vocab))
# → आप कैसे हैं?

print(translate(model, "I love you.", en_vocab, hi_vocab))
# → मैं तुमसे प्यार करता हूँ।
```

---

## Project Summary

### Dataset
- **Source:** [English-Hindi Parallel Corpus](https://www.kaggle.com/datasets/aiswaryaramachandran/hindienglish-corpora)
- **Size:** ~13,186 sentence pairs
- **Languages:** English (source) → Hindi (target)
- **Preprocessing:** Removed nulls, reset index, whitespace tokenization
- **Vocabulary:** English ~4,117 tokens | Hindi ~4,045 tokens
- **Special tokens:** `<pad>`, `<sos>`, `<eos>`, `<unk>`
- **Max sequence length:** 50 tokens

---

### Part 1 — Baseline Model

Trained a standard Transformer from scratch with fixed hyperparameters.

| Setting | Value |
|---------|-------|
| Layers | 6 encoder + 6 decoder |
| d_model | 512 |
| Attention heads | 8 |
| FFN dimension | 2048 |
| Dropout | 0.1 |
| Batch size | 60 |
| Learning rate | 1e-4 (constant) |
| Optimizer | Adam |
| Epochs | 100 |
| Final loss | 0.0991 |
| **BLEU score** | **76.98** |

Training took ~3–4 hours on a Kaggle T4 GPU.

---

### Part 2 — Hyperparameter Tuning with Ray Tune + Optuna

Used **Ray Tune** as the tuning framework and **Optuna** (TPE sampler) as the search algorithm to intelligently find optimal hyperparameters.

#### Search Space (5 hyperparameters)

| Hyperparameter | Search Space | Best Value Found |
|---------------|-------------|-----------------|
| Learning rate | log-uniform [1e-5, 1e-3] | 4.88e-04 |
| Batch size | {32, 64} | 64 |
| Attention heads | {4, 8} | 8 |
| FFN dimension | {1024, 2048} | 1024 |
| Dropout | uniform [0.1, 0.4] | 0.1507 |

#### Setup
- **Search algorithm:** Optuna TPE (Tree-structured Parzen Estimator)
- **Scheduler:** ASHA (Async Successive Halving) — kills underperforming trials early
- **Trials:** 10 total
- **Max epochs per trial:** 15
- **Grace period:** 3 epochs before pruning
- **Model during search:** 3-layer (for speed)

#### How ASHA Works
ASHA evaluates all trials at epoch 3. It keeps only the top 50% and runs them to epoch 6. Keeps top 50% again, runs to epoch 12, and so on. Bad trials are terminated early, saving significant GPU time.

---

### Part 3 — Efficiency Challenge

**Goal:** Match or beat baseline BLEU (76.98) in fewer than 100 epochs.

Retrained the full 6-layer model using the best hyperparameters found by Optuna.

| Setting | Baseline | Tuned |
|---------|----------|-------|
| Layers | 6 | 6 |
| Learning rate | 1e-4 | 1e-4 |
| Batch size | 60 | 64 |
| Attention heads | 8 | 8 |
| FFN dimension | 2048 | 1024 |
| Dropout | 0.1 | 0.1507 |
| Epochs | 100 | **50** |
| Grad clipping | ✗ | ✓ (1.0) |
| **BLEU score** | **76.98** | **89.00** |

**Result:** Achieved BLEU **89.00** in just **50 epochs** — 50% fewer epochs than the baseline, with a +12 BLEU improvement.

---

## Model Architecture

```
Transformer
├── Encoder (6 layers)
│   ├── Multi-Head Self-Attention (8 heads, d_k=64)
│   ├── Feed-Forward Network (512 → 1024 → 512)
│   ├── Layer Normalization
│   └── Dropout (0.15)
│
├── Decoder (6 layers)
│   ├── Masked Multi-Head Self-Attention
│   ├── Multi-Head Cross-Attention
│   ├── Feed-Forward Network (512 → 1024 → 512)
│   ├── Layer Normalization
│   └── Dropout (0.15)
│
└── Linear + Softmax → Hindi vocabulary (4,045 tokens)
```

**Positional Encoding:** Sinusoidal (fixed, not learned)  
**Masking:** Padding mask + causal (look-ahead) mask for decoder

---

## Sample Translations

| English | Reference Hindi | Model Output |
|---------|----------------|--------------|
| I love you. | मैं तुमसे प्यार करता हूँ। | मैं आपसे प्यार करता हूँ। |
| How are you? | आप कैसे हैं? | आप कैसे हो? |
| You should sleep. | आपको सोना चाहिए। | आपको सोना चाहिए। |
| Maybe Tom doesn't love you. | टॉम शायद तुमसे प्यार नहीं करता है। | टॉम शायद तुमसे प्यार नहीं करता है। |
| Let me tell Tom. | मुझे टॉम को बताने दीजिए। | मुझे टॉम को बताने दो। |

---

## Files

| File | Description |
|------|-------------|
| `tuned_model.pth` | PyTorch model weights (state_dict) |
| `config.json` | Architecture config + best hyperparameters |
| `en_vocab.pkl` | English vocabulary (Vocabulary object) |
| `hi_vocab.pkl` | Hindi vocabulary (Vocabulary object) |
| `inference.py` | Self-contained inference helper |

---

## Training Environment

| | Value |
|--|-------|
| Platform | Kaggle Notebooks |
| GPU | NVIDIA T4 (16GB) |
| Python | 3.12 |
| PyTorch | 2.x |
| Ray Tune | Latest |
| Optuna | Latest |

---

## Citation

```bibtex
@misc{jahanvi2026enhitransformer,
  title   = {English to Hindi Transformer with Ray Tune + Optuna},
  author  = {Jahanvi},
  year    = {2026},
  url     = {https://huggingface.co/Jahanvi16/en-to-hi-transformer}
}
```

---

## License

MIT License — free to use, modify, and distribute.
