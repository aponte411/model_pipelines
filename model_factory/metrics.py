from typing import List, Any, Dict, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch

import utils

LOGGER = utils.get_logger(__name__)


def hierarchical_macro_averaged_recall(df: pd.DataFrame) -> float:
    scores = []
    for component in [
            'grapheme_root', 'consonant_diacritic', 'vowel_diacritic'
    ]:
        y_true_subset = df[df[component] == component]['target'].values
        y_pred_subset = df[df[component] == component]['target'].values
        scores.append(
            metrics.recall_score(y_true_subset, y_pred_subset,
                                 average='macro'))
    return np.average(scores, weights=[2, 1, 1])


def macro_recall(y_true: torch.tensor,
                 preds: torch.tensor,
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
            metrics.recall_score(pred_labels[idx], y_true[:, idx])
            for idx in range(3)
        ]

    preds = _split_preds(preds)
    pred_labels = _get_labels(preds)
    recalls = _get_recalls(y_true, pred_labels)
    macro_averaged_recall = np.average(recalls, weights=[2, 1, 1])
    LOGGER.info(
        f'recall: grapheme {recalls[0]}, vowel {recalls[1]}, consonant {recalls[2]}, '
        f'total {macro_averaged_recall}, y {y_true.shape}')

    return macro_averaged_recall