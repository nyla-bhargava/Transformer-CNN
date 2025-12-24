import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_prob):
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "aupr": average_precision_score(y_true, y_prob)
    }
