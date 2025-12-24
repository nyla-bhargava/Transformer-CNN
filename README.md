### **README.md**

# Transformer–CNN for CRISPR Off-Target Prediction

This repository implements a hybrid Transformer–CNN deep learning framework for predicting CRISPR–Cas9 off-target cleavage events.  
The model integrates local mismatch-aware convolutional features with global contextual representations, and supports optional incorporation of pre-trained DNABERT embeddings.

## Abstract

Accurate prediction of CRISPR–Cas9 off-target activity is critical for safe genome editing. Existing convolutional neural network (CNN)–based approaches primarily capture local nucleotide mismatch patterns but fail to model long-range sequence dependencies. We propose a hybrid Transformer–CNN architecture that combines convolutional feature extraction with attention-based contextual modeling. Our framework optionally integrates pre-trained DNABERT embeddings to encode sgRNA sequence semantics. Experimental evaluation on proxy training data and the experimentally validated TrueOT benchmark demonstrates improved generalization compared to CNN-only baselines, particularly for high-mismatch off-target sites.

## Model Overview

**Stage 1 (Optional):**
- sgRNA sequence embeddings extracted using a pre-trained DNABERT model
- Embeddings are frozen and reused for downstream prediction

**Stage 2 (Main Model):**
- Pairwise sgRNA–off-target one-hot encoding
- Mismatch vector and PAM-distance positional encoding
- CNN blocks for local motif extraction
- Transformer encoder for long-range dependency modeling
- Feature fusion with sgRNA embeddings
- Binary off-target classification head


## Repository Structure

```

├── src/
│   ├── model.py        # Transformer–CNN architecture
│   ├── dataset.py      # Dataset and sequence encoding
│   ├── train.py        # Training pipeline
│   ├── evaluate.py     # Evaluation on TrueOT
│   └── utils.py        # Utilities and metrics
│
├── data/
│   ├── raw/            # Not included (see Dataset section)
│   └── processed/
│
├── results/
│   ├── tables/
│   └── plots/
│
└── README.md

````

## Dataset

This work uses publicly available CRISPR off-target datasets:

- **Proxy training dataset**: Used for model training and validation  
- **TrueOT dataset**: Experimentally validated benchmark for generalization evaluation

Due to licensing and size constraints, raw datasets are **not included** in this repository.  
Please place the CSV files in `data/raw/` before training.

Expected CSV columns:
- `gRNA`
- `OT`
- `label`


## Training

To train the Stage-2 Transformer–CNN model:

```bash
python src/train.py \
  --proxy_csv data/raw/Proxy_TrainCV.csv \
  --epochs 30 \
  --sg_dim 768
````

The best validation model is saved as:

```
best_stage2.pt
```

## Evaluation

To evaluate the trained model on the TrueOT benchmark:

```bash
python src/evaluate.py
```

Evaluation metrics:

* ROC-AUC
* PR-AUC (AUPR)


## Results (Summary)

| Model Variant | sgRNA Embedding | ROC-AUC | PR-AUC |
|--------------|----------------|--------|--------|
| **Stage-2 only (Transformer–CNN)** | ✗ | **0.7081** | **0.2789** |
| Stage-1 + Stage-2 | ✓ (DNABERT, frozen) | 0.6469 | 0.2553 |

(Exact values reported in the paper.)

## Notes

* Trained model checkpoints (`.pth`) are intentionally excluded from version control.
* This repository focuses on reproducibility and clarity for research use.

<img width="9606" height="5160" alt="diagram" src="https://github.com/user-attachments/assets/e00cdc1d-dcf4-4acf-8346-6f0e9e18dcbf" />

## License

This project is released for academic and research use only.

## 2.3 Save, commit, push

```bash
git add README.md
git commit -m "Add research-grade README"
git push origin main
````

## Reproducibility

All experiments are controlled via YAML configuration files in `experiments/`.

To reproduce evaluation metrics and plots on TrueOT:

```bash
python src/reproduce_results.py --config experiments/transformer_cnn.yaml


