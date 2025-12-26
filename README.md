# Transformer–CNN Hybrid Framework for CRISPR Off-Target Prediction

This repository contains the official implementation of a **two-stage deep learning framework** for CRISPR–Cas9 off-target prediction using **DNABERT embeddings** and **CNN / Transformer architectures**.

The framework is designed with **strict reproducibility** and **student research paper standards** in mind.

## Abstract

Accurate prediction of CRISPR–Cas9 off-target activity is critical for safe and effective genome editing. Traditional CNN-based models effectively capture local sequence motifs but lack mechanisms for modeling long-range dependencies. In this work, we introduce a **Transformer–CNN hybrid framework** that leverages the strengths of both self-attention and convolutional operations to predict off-target effects from sgRNA–DNA sequence pairs. Evaluation on the **TrueOT benchmark** demonstrates improved generalization compared to CNN-only and Transformer-only baselines.

## Model Overview

### Stage 1 (Optional)
- sgRNA sequence embeddings extracted using a pre-trained **DNABERT** model
- DNABERT parameters are **frozen** and reused for downstream prediction

### Stage 2 (Main Model)
- Pairwise sgRNA–off-target sequence encoding
- Mismatch vector and PAM-distance positional encoding
- CNN blocks for local motif extraction
- Transformer encoder for long-range dependency modeling
- Feature fusion with sgRNA embeddings
- Binary classification head for off-target prediction

## Repository Structure

```
Transformer-CNN/
│
├── data/
│ ├── raw/ # Original datasets (user-provided)
│ └── processed/ # DNABERT-encoded features
│
├── src/
│ ├── preprocess.py # DNABERT feature extraction
│ ├── cnn_only.py # CNN-only baseline (DNABERT)
│ ├── transformer_only.py # Transformer-only baseline (DNABERT)
│ ├── hybrid_model.py # Stage-2 Transformer–CNN hybrid
│ └── reproduce_results.py # End-to-end reproducibility script
│
├── requirements.txt
└── README.md

````
## Data Availability

Due to licensing restrictions, datasets are **not included** in this repository.

To reproduce the experiments, place the following files in `data/raw/`:

- `Proxy_TrainCV.csv`
- `Proxy_Validation.csv`
- `TrueOT_1806uniqueTriplet_gRNA_OT_label.csv`

### Required CSV Schema

Each dataset must contain:

gRNA   : nucleotide sequence (string)
label  : off-target label (0 or 1)

## Environment Setup

**Recommended platform:** Google Colab  

This project is compatible with modern PyTorch and Transformers.

### Verified versions

This project is compatible with modern PyTorch and Transformers.

- torch 2.9.0
- transformers 4.57.3
- numpy 2.0.2
- pandas 2.2.2
- scikit-learn 1.6.1

Install dependencies:
```bash
pip install -r requirements.txt
````

## Step 1 — Preprocess Data (DNABERT Encoding)

Generate DNABERT embeddings from raw sequences:
```bash
python src/preprocess.py
````
This will create the following files in data/processed/:

- proxy_train_encoded.npz
- proxy_val_encoded.npz
- trueot_encoded.npz

Each .npz file contains:
- X: pooled DNABERT embeddings
- y: binary labels

## Step 2 — Train Baseline Models
**CNN-only (DNABERT)**
```bash
python src/cnn_only.py
````
**Transformer-only (DNABERT)**
```bash
python src/transformer_only.py
````
These models serve as ablation baselines for the hybrid architecture. 

## Step 3 — Train Hybrid Transformer–CNN Model
```bash
python src/hybrid_model.py
````
This model:

Uses frozen DNABERT embeddings (Stage-1)

Applies Transformer encoders + CNN feature extraction (Stage-2)

## Step 4 — Reproduce All Results

To reproduce all reported metrics in a single command:
```bash
python src/reproduce_results.py
````
This script:
- Loads processed datasets
- Trains all models
- Evaluates on the TrueOT benchmark
- Reports ROC-AUC scores

## Model Architecture

<img width="9606" height="5160" alt="diagram" src="https://github.com/user-attachments/assets/e00cdc1d-dcf4-4acf-8346-6f0e9e18dcbf" />


**Figure: Overview of the Transformer–CNN hybrid architecture.**

- Input Encoding: Paired sgRNA–DNA sequences (one-hot encoded, 4 × 23)
- Transformer Encoder: Captures long-range contextual dependencies
- 1D CNN Block: Extracts local mismatch patterns
- Feature Fusion: Concatenation of global and local representations
- Fully Connected Layers: Dense prediction head with dropout
- Output: Sigmoid probability of off-target activity

Ablation Variants:

- CNN-only: Transformer branch removed
- Transformer-only: CNN branch removed

### Ablation Implementation

CNN-only and Transformer-only baselines are implemented by selectively
disabling architectural components within `model.py`.

- CNN-only: Transformer encoder disabled
- Transformer-only: CNN branch disabled
- Hybrid: Both branches enabled

Model variants are selected via configuration flags passed to `train.py`
and are executed sequentially in `reproduce_results.py`.
## 
**Evaluation Metric**

- All models are evaluated using:
- ROC–AUC on the TrueOT dataset
This metric is standard in CRISPR off-target prediction literature.

**Reproducibility Notes**

- DNABERT weights are frozen
- Random seeds are fixed
- No external preprocessing is required
All experiments run on CPU or GPU

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
| CNN-only (ours) | CNN | Proxy datasets | *reported in paper* |
| Transformer-only (ours) | Transformer | Proxy datasets | *reported in paper* |
| **Transformer–CNN (ours)** | Hybrid | Proxy datasets | **0.7081** |

(Exact values reported in the paper.)

## Notes

* Trained model checkpoints (`.pth`) are intentionally excluded from version control.
* This repository focuses on reproducibility and clarity for research use.

## License

This project is released for academic and research use only.

## Citation

If you use this work, please cite:
```bash
@article
````







