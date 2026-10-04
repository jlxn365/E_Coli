import numpy as np
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1) Logistic Regression: chi2 -> SVD -> scaler -> LR
# ---------------------------------------------------------------------------
def build_lr_pipeline(cfg: dict, n_features: int, random_state: int) -> Pipeline:
    k = min(cfg["k"], n_features)
    n_components = min(cfg["svd_components"], k - 1)
    return Pipeline(
        [
            ("selector", SelectKBest(chi2, k=k)),  # feature selection
            ("svd", TruncatedSVD(n_components=n_components, random_state=random_state)),
            ("scaler", StandardScaler()),  # standardize SVD output
            (
                "lr",
                LogisticRegression(
                    penalty="l2",
                    solver="saga",
                    class_weight="balanced",
                    C=cfg["C"],
                    max_iter=cfg["max_iter"],
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 2) LightGBM: chi2 band (skip top) -> LGBM importance ranking -> final LGBM
#    Everything happens inside fit(), so cross-validation has no leakage.
# ---------------------------------------------------------------------------
class LGBMChiSelectClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        skip_top=10_000,
        k_filter=80_000,
        k_final=15_000,
        ranker_params=None,
        final_params=None,
        random_state=42,
    ):
        self.skip_top = skip_top
        self.k_filter = k_filter
        self.k_final = k_final
        self.ranker_params = ranker_params
        self.final_params = final_params
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        chi_scores, _ = chi2(X, y)
        chi_scores = np.nan_to_num(chi_scores, nan=0.0)
        chi_order = np.argsort(chi_scores)[::-1]  # most -> least informative

        skip = self.skip_top if self.skip_top < n_features // 2 else 0
        candidate_idx = chi_order[skip : skip + self.k_filter]

        ranker = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            class_weight="balanced",
            max_depth=-1,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=-1,
            **(self.ranker_params or {}),
        )
        ranker.fit(X[:, candidate_idx], y)

        sorted_local_idx = np.argsort(ranker.feature_importances_)[::-1]
        top_local = sorted_local_idx[: self.k_final]
        self.feat_idx_ = np.unique(candidate_idx[top_local])

        self.model_ = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            class_weight="balanced",
            max_depth=-1,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=-1,
            **(self.final_params or {}),
        )
        self.model_.fit(X[:, self.feat_idx_], y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X[:, self.feat_idx_])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# 3) XGBoost: chi2 band (skip top) -> gain importance -> top-K features -> average of several seeds
# ---------------------------------------------------------------------------
class XGBSeedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        params=None,
        num_boost_round=334,
        k_chi2=100_000,
        k_final=10_000,
        seeds=None,
    ):
        self.params = params
        self.num_boost_round = num_boost_round
        self.k_chi2 = k_chi2
        self.k_final = k_final
        self.seeds = seeds

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        params = {"objective": "binary:logistic", **(self.params or {})}

        self.chi2_selector_ = SelectKBest(
            chi2, k=min(self.k_chi2, X.shape[1])
        ).fit(X, y)
        X_chi2 = self.chi2_selector_.transform(X)
        chi2_idx = self.chi2_selector_.get_support(indices=True)

        bst = xgb.train(
            params, xgb.DMatrix(X_chi2, label=y), num_boost_round=self.num_boost_round
        )
        importance_dict = bst.get_score(importance_type="gain")
        importance_scores = np.zeros(X_chi2.shape[1], dtype=np.float32)
        for f, score in importance_dict.items():
            importance_scores[int(f[1:])] = score  # remove "f"
        ranked_local_idx = np.argsort(-importance_scores)
        top_local = ranked_local_idx[: min(self.k_final, X_chi2.shape[1])]

        self.feat_idx_ = chi2_idx[top_local]

        dtrain = xgb.DMatrix(X[:, self.feat_idx_], label=y)
        self.models_ = []
        for s in self.seeds or [0, 1, 2, 3, 4]:
            p = {**params, "seed": s}
            self.models_.append(
                xgb.train(p, dtrain, num_boost_round=self.num_boost_round)
            )
        return self

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X[:, self.feat_idx_])
        preds = np.column_stack([m.predict(dtest) for m in self.models_])
        p = preds.mean(axis=1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
#Optuna search for the XGBoost params
# ---------------------------------------------------------------------------
def tune_xgb_optuna(X, y, n_trials=25, n_splits=5, random_state=42, device="cuda"):
    import optuna

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": device,
            "n_jobs": -1,
        }
        num_boost_round = trial.suggest_int("num_boost_round", 50, 500)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        f1_scores = []
        for train_idx, val_idx in skf.split(X, y):
            dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
            dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, "valid")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )
            preds = bst.predict(dval)
            f1_scores.append(
                f1_score(y[val_idx], (preds > 0.5).astype(int), average="macro", zero_division=0)
            )
        return np.mean(f1_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best F1: {study.best_value}")
    print("Best hyperparameters:", study.best_params)
    return study.best_params
