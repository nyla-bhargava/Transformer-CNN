# CRISPR Off-Target Prediction Using a Transformer–CNN Hybrid Model

This repository contains the official implementation of a Transformer–CNN hybrid
deep learning framework for CRISPR–Cas9 off-target site prediction. The model is
designed to learn mismatch-aware local sequence representations and is evaluated
for cross-dataset generalization on the experimentally validated **TrueOT**
benchmark.

The work focuses on robust generalization from proxy datasets to real biological
off-target sites.

## 1. Overview

CRISPR off-target prediction remains challenging due to domain shifts between
high-throughput proxy datasets and experimentally validated off-target sites.
This repository presents a neural architecture that explicitly models
sgRNA–off-target mismatches using a combination of convolutional and transformer
layers.

The proposed approach emphasizes:
- Explicit pairwise encoding of sgRNA and off-target sequences
- Mismatch-aware sequence representations
- Local pattern extraction using convolutional layers
- Long-range dependency modeling using transformer encoders
- Evaluation on TrueOT for realistic generalization assessment

## 2. Model Architecture

### Stage-2 Model (Final)

The final model operates directly on sgRNA–off-target pairs using:
- Pairwise one-hot sequence encoding
- Explicit mismatch vectors and mismatch summary features
- Convolutional layers for local feature extraction
- Transformer encoder layers for contextual modeling
- A fully connected classifier for off-target probability estimation

This Stage-2-only model demonstrates superior generalization performance on
TrueOT and constitutes the primary contribution of this work.

### Stage-1 Model (Ablation Only)

A pretrained DNABERT-based encoder is optionally incorporated as Stage-1 to
provide sequence-level embeddings of sgRNAs. This component is included solely
for ablation and reproducibility analysis and is not used in the final model
reported in the paper.

## 3. Experimental Results

### TrueOT Generalization Performance

| Model Variant | ROC-AUC | AUPR |
|--------------|---------|------|
| Stage-2 only (Final Model) | 0.7081 | 0.2789 |
| Stage-1 + Stage-2 | 0.6469 | 0.2553 |

The results indicate that explicit mismatch-aware local representations
generalize more effectively to experimentally validated off-target sites than
pretrained sequence embeddings.

## 4. Repository Structure
CRISPR-OffTarget-Transformer/

│
├── code/
│ └── research.py
│
├── models/
│ ├── best_stage2.pt
│ ├── stage2_with_stage1.pth
│ ├── sg_embeddings.pt
│ └── stage1/
│ ├── stage1_dnabert_finetuned.pt
│ └── tokenizer/
│ ├── vocab.txt
│ ├── tokenizer.json
│ ├── tokenizer_config.json
│ └── special_tokens_map.json
│
├── results/
│ └── TrueOT_Results_Table.docx
│
├── README.md
└── requirements.txt

## 5. Datasets

This work uses publicly available datasets:
- Proxy datasets for training
- The TrueOT dataset for cross-dataset generalization evaluation

Due to licensing and redistribution restrictions, datasets are not included in
this repository. Users should obtain them from the original sources.

## 6. Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt

Core dependencies

PyTorch

HuggingFace Transformers

scikit-learn

NumPy

Pandas

einops

matplotlib

## 7. Usage

To train and evaluate the model, run:

python code/stage2 CNN.py


To switch between ablation settings, modify the configuration variable:

USE_STAGE1 = True  # or False


The final results reported in the paper correspond to USE_STAGE1 = False.

8. Reproducibility Notes

All models are evaluated exclusively on the TrueOT benchmark

Identical training protocols are used across ablation settings

No hyperparameter tuning is performed on the TrueOT dataset

Results are reported from fixed, single-run experiments

9. Code and Model Availability

The source code and trained model checkpoints are publicly available in this
repository to facilitate reproducibility and further research.

10. Citation

If you use this code or models in your research, please cite:

Manuscript under preparation


A BibTeX entry will be provided upon publication.

11. Contact

For questions, issues, or suggestions, please open an issue in this repository.

