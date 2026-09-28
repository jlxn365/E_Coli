import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Out-of-fold predictions
# ---------------------------------------------------------------------------
def get_oof_predictions(clf, X, y, n_splits=5, random_state=42, model_name="MODEL"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y))
    fold_indices = np.zeros(len(y), dtype=int)  # 0-based for every model

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"[{model_name}] Fold {fold+1}/{n_splits}")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = clone(clf)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        oof_pred[val_idx] = proba
        fold_indices[val_idx] = fold

        f1 = f1_score(y_val, (proba >= 0.5).astype(int), average="macro")
        print(f"[{model_name}] Fold F1 = {f1:.4f}")

    overall_f1 = f1_score(y, (oof_pred >= 0.5).astype(int), average="macro")
    print(f"[{model_name}] OOF Macro-F1 = {overall_f1:.4f}")
    return oof_pred, fold_indices


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
def best_f1_threshold(y_true, proba, n_steps=2001):
    thresholds = np.linspace(0, 1, n_steps)
    f1s = [f1_score(y_true, (proba >= t).astype(int), average="macro") for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return thresholds[best_idx], f1s[best_idx]


def fold_median_threshold(y_true, proba, fold_idx, n_splits=5, n_steps=2001):
    thresholds = np.linspace(0, 1, n_steps)
    fold_best = []
    for fold in range(n_splits):
        mask = fold_idx == fold
        y_f, p_f = y_true[mask], proba[mask]
        f1s = [f1_score(y_f, (p_f >= t).astype(int), average="macro") for t in thresholds]
        fold_best.append(thresholds[np.argmax(f1s)])
    return np.median(fold_best)


# ---------------------------------------------------------------------------
# Meta features
# ---------------------------------------------------------------------------
def calibrate(p):
    return 1 / (1 + np.exp(-3 * (p - 0.5)))


def build_meta_features(p_lr, p_xgb, p_lgb):
    """Same function for train (OOF) and test, so both sides always match."""
    p_lr_c, p_xgb_c, p_lgb_c = calibrate(p_lr), calibrate(p_xgb), calibrate(p_lgb)
    return np.column_stack(
        [
            p_lr, p_xgb, p_lgb,
            p_lr_c, p_xgb_c, p_lgb_c,
            p_xgb * p_lgb,
            p_lr * p_lgb,
            p_lr * p_xgb,
        ]
    )


def _meta_model(C):
    return LogisticRegression(C=C, solver="lbfgs", max_iter=5000, class_weight="balanced")


def meta_oof_predictions(Z, y, C, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y))
    fold_idx = np.zeros(len(y), dtype=int)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(Z, y)):
        model = _meta_model(C).fit(Z[tr_idx], y[tr_idx])
        oof[val_idx] = model.predict_proba(Z[val_idx])[:, 1]
        fold_idx[val_idx] = fold
    return oof, fold_idx


def select_meta_C_and_threshold(Z, y, C_grid, n_splits=5, random_state=42):
    """One C is chosen; the OOF, the threshold and the final model all use it."""
    best = None
    print("========== META SUMMARY ==========")
    for C in C_grid:
        oof, fold_idx = meta_oof_predictions(Z, y, C, n_splits, random_state)
        thr = fold_median_threshold(y, oof, fold_idx, n_splits)
        f1 = f1_score(y, (oof >= thr).astype(int), average="macro")
        print(f"C={C:<5} thr={thr:.4f}  OOF-F1={f1:.5f}")
        if best is None or f1 > best["f1"]:
            best = {"C": C, "thr": thr, "f1": f1}
    print(f"BEST C: {best['C']}  THRESHOLD: {best['thr']:.4f}  OOF-F1: {best['f1']:.5f}")
    return best["C"], best["thr"]


def fit_meta(Z_train, y_train, C):
    return _meta_model(C).fit(Z_train, y_train)
