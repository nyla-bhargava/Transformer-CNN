# CNN-only Ablation (DNABERT)

## Overview
This directory contains results for the **CNN-only ablation** of the proposed
Transformer–CNN framework for CRISPR off-target prediction.

The CNN-only model is designed to **isolate the contribution of convolutional
inductive bias** by removing the Transformer module while keeping all other
experimental factors constant.

## Model Description
- Input representation: Frozen DNABERT-2-117M embeddings
- Pooling: Masked mean pooling over valid tokens
- Architecture: Feedforward convolutional classifier
- Training objective: Binary classification (off-target vs non-off-target)

The model does **not** include self-attention or sequence-level contextual modeling.

## Training and Evaluation Protocol
- Training data: Proxy CRISPR off-target datasets
- Validation: Proxy validation split (used for checkpoint selection)
- Evaluation dataset: TrueOT benchmark
- Loss function: Class-weighted BCEWithLogitsLoss
- Evaluation metric: ROC-AUC

TrueOT is used **only for final evaluation** and is never seen during training
or validation.

## Results
Final performance on the TrueOT benchmark is reported in `metrics.csv`.

These results demonstrate that **convolutional modeling alone is insufficient**
to achieve robust generalization across experimental conditions, motivating
the need for hybrid architectures.

## Notes
- Model checkpoints and intermediate embeddings are not included.
- All results are reproducible using the training scripts provided in `src/`.
