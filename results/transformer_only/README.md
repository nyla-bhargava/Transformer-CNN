# Transformer-only Ablation (DNABERT)

## Overview
This directory contains results for the **Transformer-only ablation** of the
proposed Transformer–CNN framework for CRISPR off-target prediction.

The Transformer-only model isolates the effect of **self-attention–based global
sequence modeling** by removing the convolutional component.

## Model Description
- Input representation: Frozen DNABERT-2-117M embeddings
- Pooling: Masked mean pooling over valid tokens
- Architecture: Transformer encoder followed by linear classification
- Training objective: Binary classification (off-target vs non-off-target)

No convolutional layers are used in this ablation.

## Training and Evaluation Protocol
- Training data: Proxy CRISPR off-target datasets
- Validation: Proxy validation split (used for checkpoint selection)
- Evaluation dataset: TrueOT benchmark
- Loss function: Class-weighted BCEWithLogitsLoss
- Evaluation metric: ROC-AUC

The evaluation strictly follows the TrueOT protocol to assess cross-dataset
generalization.

## Results
Final performance on the TrueOT benchmark is summarized in `metrics.csv`.

While the Transformer-only model improves generalization compared to the
CNN-only baseline, it remains inferior to the proposed hybrid architecture,
highlighting the complementary roles of global attention and local feature
extraction.

## Notes
- Trained model weights are not included in this repository.
- Experiments can be reproduced using the provided scripts and configurations.
