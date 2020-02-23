import numpy as np
import pandas as pd
import sklearn.metrics


def hierarchical_macro_averaged_recall(df: pd.DataFrame) -> float:
    scores = []
    for component in [
            'grapheme_root', 'consonant_diacritic', 'vowel_diacritic'
    ]:
        y_true_subset = df[df[component] == component]['target'].values
        y_pred_subset = df[df[component] == component]['target'].values
        scores.append(
            sklearn.metrics.recall_score(y_true_subset,
                                         y_pred_subset,
                                         average='macro'))
    return np.average(scores, weights=[2, 1, 1])
