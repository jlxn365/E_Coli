from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from data import load_params, load_test_npz, load_train_npz
from features import reduce_input
from models import LGBMChiSelectClassifier, XGBSeedEnsemble, build_lr_pipeline
from stacking import (
    build_meta_features,
    fit_meta,
    get_oof_predictions,
    select_meta_C_and_threshold,
)


def run_base_model(name, model, X_train, y_train, X_test, params):
    """OOF predictions on train + predictions on test (refit on the full train).
    Results are cached: delete the cache folder if you change a config."""
    cache = Path(params["cache_dir"]) / f"{name}.npz"
    if cache.exists():
        print(f"[{name}] loaded from cache")
        d = np.load(cache)
        return d["oof"], d["test"]

    oof, _ = get_oof_predictions(
        model, X_train, y_train, params["n_splits"], params["random_state"], name
    )
    full_model = clone(model).fit(X_train, y_train)
    test_proba = full_model.predict_proba(X_test)[:, 1]

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, oof=oof, test=test_proba)
    return oof, test_proba


def main():
    params = load_params("./params.yaml")
    rs = params["random_state"]
    data_path = Path(params["data_dir"])

    #Load data
    X_train, y_train, _ = load_train_npz(data_path / "train.npz")
    X_test, ids_test = load_test_npz(data_path / "test.npz")

    #Frequency filter (columns chosen on train only)
    X_train, X_test = reduce_input(
        X_train, X_test, params["freq_higher_bound"], params["freq_lower_bound"]
    )
    n_features = X_train.shape[1]

    #Base models (feature selection happens inside each fold)
    lr_model = build_lr_pipeline(params["lr"], n_features, rs)
    lgb_model = LGBMChiSelectClassifier(
        skip_top=params["lgbm"]["skip_top"],
        k_filter=params["lgbm"]["k_filter"],
        k_final=params["lgbm"]["k_final"],
        ranker_params=params["lgbm"]["ranker_params"],
        final_params=params["lgbm"]["final_params"],
        random_state=rs,
    )
    xgb_model = XGBSeedEnsemble(
        params=params["xgb"]["params"],
        num_boost_round=params["xgb"]["num_boost_round"],
        k_final=params["xgb"]["k_final"],
        seeds=params["xgb"]["seeds"],
    )

    lr_oof, lr_test = run_base_model("LR", lr_model, X_train, y_train, X_test, params)
    lgb_oof, lgb_test = run_base_model("LGBM", lgb_model, X_train, y_train, X_test, params)
    xgb_oof, xgb_test = run_base_model("XGB", xgb_model, X_train, y_train, X_test, params)

    #Stacking
    Z_train = build_meta_features(lr_oof, xgb_oof, lgb_oof)
    Z_test = build_meta_features(lr_test, xgb_test, lgb_test)
    print("Z_train shape:", Z_train.shape)
    print("Z_test shape:", Z_test.shape)

    best_C, threshold = select_meta_C_and_threshold(
        Z_train, y_train, params["stacking"]["C_grid"], params["n_splits"], rs
    )
    final_meta = fit_meta(Z_train, y_train, best_C)
    meta_test_proba = final_meta.predict_proba(Z_test)[:, 1]
    final_labels = (meta_test_proba >= threshold).astype(int)

    #Submission
    submission = pd.DataFrame({"id": ids_test, "label": final_labels})
    print("Number of predicted 1:", int((submission["label"] == 1).sum()))
    submission.to_csv("submission_stacking.csv", index=False)
    print("Saved: submission_stacking.csv")


if __name__ == "__main__":
    main()
