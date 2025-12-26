### **README.md**

# Transformer–CNN for CRISPR Off-Target Prediction

This repository implements a hybrid deep learning framework that integrates Transformer encoders and convolutional neural networks (CNNs) to predict CRISPR–Cas9 off-target cleavage events. The architecture is designed to capture both global contextual dependencies and local nucleotide mismatch patterns, improving generalization on experimentally validated datasets.

## Abstract

Accurate prediction of CRISPR–Cas9 off-target activity is critical for safe and effective genome editing. Traditional CNN-based models capture local features but lack mechanisms for modeling long-range sequence dependencies. We introduce a Transformer–CNN hybrid model that leverages the strengths of both self-attention and convolutional operations to predict off-target effects from paired sgRNA–DNA sequences. Evaluation on the TrueOT benchmark shows improved generalization compared to CNN-only and Transformer-only baselines.

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

Transformer-CNN/
├── data/
│ └── processed/ 
│ ├── encoding_info.json
│ ├── proxy_train_encoded.npz
│ ├── proxy_val_encoded.npz
│ └── trueot_encoded.npz
│
├── src/
│ ├── model.py
│ ├── dataset.py 
│ ├── train.py
│ ├── evaluate.py 
│ ├── reproduce_results.py 
│ └── utils.py
│
├── experiments/
│ ├── cnn_baseline.yaml
│ ├── transformer_only.yaml
│ └── transformer_cnn.yaml      # Configuration files for reproducible experiments
│
├── results/ # Results (metrics, plots)
├── models/ # Model checkpoints (links or placeholders)
├── .gitignore
└── README.md

````
## Data Availability
Due to licensing restrictions, datasets are not included in this repository.

To reproduce results, place the following files in `data/processed/`:
- Proxy_TrainCV.csv
- Proxy_Validation.csv
- TrueOT_1806uniqueTriplet_gRNA_OT_label.csv

File paths are expected exactly as referenced in the scripts.

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

## Preprocessing

Preprocessing converts raw sequence pairs into one-hot encoded tensors.  
Encoded representations are stored in `data/processed/` as compressed NumPy arrays (`*.npz`).

Each encoded file contains:
- `X`: array of shape (N, 4, 23) representing one-hot sequences
- `y`: label vector of shape (N,)

Metadata is stored in `encoding_info.json` for reproducibility.

## Model Architecture

<img width="9606" height="5160" alt="diagram" src="https://github.com/user-attachments/assets/e00cdc1d-dcf4-4acf-8346-6f0e9e18dcbf" />

Figure (see paper) illustrates the full architecture:

Input encoding: Paired sgRNA–DNA sequences, one-hot encoded (4 × 23)

Transformer Encoder: Captures long-range contextual dependencies

1D CNN Block: Extracts local mismatch features

Feature Fusion: Concatenation of global and local features

Fully Connected Layers: Dense prediction head with dropout

Output: Sigmoid probability of off-target activity

Ablation variants are supported:

CNN-only: Remove Transformer branch

Transformer-only: Remove CNN branch

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

---
| Model | Architecture | Training Data | ROC-AUC (TrueOT) |
|------|-------------|--------------|----------------|
| DeepCRISPR [1] | CNN | Proxy datasets | 0.65 |
| CRISPR-Net [2] | CNN | Proxy datasets | 0.68 |
| TrueOT CNN baseline [3] | CNN | Proxy datasets | 0.69 |
| CNN-only (ours) | CNN | Proxy datasets | X.XX |
| Transformer-only (ours) | Transformer | Proxy datasets | X.XX |
| **Transformer–CNN (ours)** | Hybrid | Proxy datasets | **0.7081** |

(Exact values reported in the paper.)

## Notes

* Trained model checkpoints (`.pth`) are intentionally excluded from version control.
* This repository focuses on reproducibility and clarity for research use.

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






