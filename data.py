from pathlib import Path
from typing import Any, Tuple

import numpy as np
import yaml
from scipy import sparse


def load_params(path: str = "./params.yaml") -> Any:
    """Load the YAML parameters for training

    Args:
        path (str, optional): Parameter path. Defaults to ./params.yaml.

    Returns:
        Any: Parameters usually in dict format
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _unwrap(X):
    """Some .npz files store a sparse matrix as a 0-d object array: unwrap it."""
    if isinstance(X, np.ndarray) and X.dtype == object and X.ndim == 0:
        X = X.item()
    if sparse.issparse(X):
        X = X.tocsr()
    return X


def load_train_npz(filepath: str) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Load training set

    Args:
        filepath (str): File name

    Returns:
        Tuple: Training data, labels and ids
    """
    data = np.load(filepath, allow_pickle=True)
    return _unwrap(data["X_train"]), np.asarray(data["y_train"]).astype(int), data["ids"]


def load_test_npz(filepath: str) -> Tuple[Any, np.ndarray]:
    """Load testing set

    Args:
        filepath (str): File name

    Returns:
        Tuple: Testing data and ids
    """
    data = np.load(filepath, allow_pickle=True)
    return _unwrap(data["X_test"]), data["ids"]
