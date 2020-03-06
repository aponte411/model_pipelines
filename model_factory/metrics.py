from typing import List, Any, Dict, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch

import utils

LOGGER = utils.get_logger(__name__)



def macro_recall(preds: torch.tensor,
                 y_true: torch.tensor,
                 n_grapheme: int = 168,
                 n_vowel: int = 11,
                 n_consonant: int = 7) -> float:
    def _split_preds(
            preds: torch.tensor,
            shape: List = [n_grapheme, n_vowel, n_consonant]) -> torch.tensor:
        return torch.split(preds, shape, dim=1)

    def _get_labels(preds: torch.tensor) -> List:
        return [torch.argmax(label, dim=1).cpu().numpy() for label in preds]

    def _get_recalls(y_true: torch.tensor, pred_labels: List) -> List[float]:
        y_true = y_true.cpu().numpy()
        return [
            metrics.recall_score(pred_labels[idx],
                                 y_true[:, idx],
                                 average='macro') for idx in range(3)
        ]

    preds = _split_preds(preds)
    pred_labels = _get_labels(preds)
    recalls = _get_recalls(y_true, pred_labels)
    macro_averaged_recall = np.average(recalls, weights=[2, 1, 1])
    LOGGER.info(f'Recalls: Grapheme {recalls[0]:.3f}, Vowel {recalls[1]:.3f}, Consonant {recalls[2]:.3f}')
    LOGGER.info(f'Hierarchical Macro-Averaged Recall: {macro_averaged_recall})')
    return macro_averaged_recall