from typing import Tuple

import numpy as np


def frequency_based_reduction(
    X, higher_freq_bound: float = 0.97, lower_freq_bound: float = 0.01
) -> np.ndarray:
    """Select columns that are not extremely rare or extremely common.

    The selection only uses X (no labels), and is computed on the train set only.

    Args:
        X: Dataset (dense or sparse)
        higher_freq_bound (float, optional): Max fraction of non-zero rows. Defaults to 0.97.
        lower_freq_bound (float, optional): Min fraction of non-zero rows. Defaults to 0.01.

    Returns:
        np.ndarray: Indices of the selected columns
    """
    non_zero_counts = np.asarray((X != 0).sum(axis=0)).ravel()
    min_threshold = X.shape[0] * lower_freq_bound
    max_threshold = X.shape[0] * higher_freq_bound
    selected_cols = np.where(
        (non_zero_counts >= min_threshold) & (non_zero_counts <= max_threshold)
    )[0]
    return selected_cols


def reduce_input(X_train, X_test, higher_freq_bound: float, lower_freq_bound: float) -> Tuple:
    """Reduce the input training and testing set (original columns are kept).

    Returns:
        Tuple: The reduced train and test sets
    """
    selected_columns = frequency_based_reduction(X_train, higher_freq_bound, lower_freq_bound)
    print(f"Frequency filter: {X_train.shape[1]} -> {len(selected_columns)} features")
    return X_train[:, selected_columns], X_test[:, selected_columns]
