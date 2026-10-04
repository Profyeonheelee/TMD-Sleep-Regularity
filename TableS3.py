"""Supplementary Table S3: cross-validated predictive performance of five learners.

Repeated 5-fold cross-validation (2 repeats). Out-of-fold predictions are saved for
Supplementary Figures S2 and S3.
"""

import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from common import RESULTS, load_data, fdr
from gated_transformer import GatedFTTClassifier, GatedFTTRegressor

FEATURES = [
    "Non_regular",
    "Schedule_range_h",
    "Midsleep_h",
    "Bedtime_h",
    "Sleep_duration_h",
    "PSQI_C1",
    "PSQI_C2",
    "PSQI_C4",
    "PSQI_C5",
    "PSQI_C6",
    "PSQI_C7",
    "STOP_score",
    "Age",
    "Female",
    "BMI",
    "log_dur",
    "Bilateral_pain",
    "Stress",
    "Bruxism",
    "Clenching",
    "Contributing_factors",
    "Systemic_disease",
    "Entry_period",
]
MISSING_INDICATORS = ["BMI", "Midsleep_h", "Bedtime_h", "Sleep_duration_h", "PSQI_C4", "Schedule_range_h"]
LEARNERS = ["Lasso", "GBM", "DNN", "Gated Transformer"]
ALL = LEARNERS + ["Ensemble"]
OUTCOMES = [("VAS", False), ("DI", False), ("CMI", False), ("Locking", True)]
K, R = 5, 2
CV_FILE = RESULTS / "cv_predictions.pkl"


def features(df):
    X = df[FEATURES].astype(float).copy()
    for c in MISSING_INDICATORS:
        X["miss_" + c] = X[c].isna().astype(float)
    return X


def learner(name, binary):
    imp = SimpleImputer(strategy="median")
    if name == "Gated Transformer":
        return make_pipeline(imp, GatedFTTClassifier() if binary else GatedFTTRegressor())
    if name == "Lasso":
        est = LogisticRegressionCV(cv=3, Cs=10, max_iter=3000) if binary else LassoCV(cv=3, n_alphas=30, max_iter=5000)
        return make_pipeline(imp, StandardScaler(), est)
    if name == "GBM":
        cls = HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor
        return make_pipeline(
            imp,
            cls(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=0,
            ),
        )
    cls = MLPClassifier if binary else MLPRegressor
    return make_pipeline(
        imp,
        StandardScaler(),
        cls(
            hidden_layer_sizes=(128, 64, 32),
            alpha=1e-2,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            max_iter=500,
            random_state=0,
        ),
    )


def score(y, p, binary):
    return roc_auc_score(y, p) if binary else r2_score(y, p)


def cross_validate(d):
    out = {}
    for y, binary in OUTCOMES:
        df = d[d[y].notna()].reset_index(drop=True)
        X, Y = features(df), df[y].values.astype(float)
        splitter = (
            RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=7)
            if binary
            else RepeatedKFold(n_splits=K, n_repeats=R, random_state=7)
        )
        folds = {k: [] for k in ALL}
        oof = {k: np.zeros((R, len(Y))) for k in ALL}
        for i, (tr, te) in enumerate(splitter.split(X, Y if binary else None)):
            P = {}
            for k in LEARNERS:
                m = learner(k, binary).fit(X.iloc[tr], Y[tr])
                P[k] = m.predict_proba(X.iloc[te])[:, 1] if binary else m.predict(X.iloc[te])
            P["Ensemble"] = np.mean([P[k] for k in LEARNERS], axis=0)
            for k, p in P.items():
                folds[k].append(score(Y[te], p, binary))
                oof[k][i // K, te] = p
        out[y] = {"folds": folds, "oof": oof, "Y": Y, "n_train": len(Y) * (K - 1) // K, "n_test": len(Y) // K}
        print(y, {k: round(np.mean(v), 3) for k, v in folds.items()}, flush=True)
    return out


def corrected_t(a, b, n_train, n_test):
    """Corrected resampled t test for repeated k-fold CV (Nadeau & Bengio 2003; Bouckaert & Frank 2004)."""
    diff = np.array(a) - np.array(b)
    v = diff.var(ddof=1)
    t = diff.mean() / np.sqrt((1 / (K * R) + n_test / n_train) * v) if v > 0 else 0.0
    return 2 * stats.t.sf(abs(t), K * R - 1)


def delong(y, p1, p2):
    """DeLong test for two correlated AUCs."""
    y = np.asarray(y).astype(int)

    def components(p):
        pos, neg = p[y == 1], p[y == 0]
        v10 = np.array([(np.sum(x > neg) + 0.5 * np.sum(x == neg)) / len(neg) for x in pos])
        v01 = np.array([(np.sum(pos > x) + 0.5 * np.sum(pos == x)) / len(pos) for x in neg])
        return v10.mean(), v10, v01

    a1, v10a, v01a = components(p1)
    a2, v10b, v01b = components(p2)
    S = np.cov(np.vstack([v10a, v10b])) / len(v10a) + np.cov(np.vstack([v01a, v01b])) / len(v01a)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return 2 * stats.norm.sf(abs(z))


def summary(cv):
    rows = []
    for y, binary in OUTCOMES:
        r = cv[y]
        means = {k: np.mean(r["folds"][k]) for k in ALL}
        best = max(means, key=means.get)
        friedman = stats.friedmanchisquare(*[r["folds"][k] for k in ALL]).pvalue
        others = [k for k in ALL if k != best]
        if binary:
            raw = [delong(r["Y"], r["oof"][best][0], r["oof"][k][0]) for k in others]
        else:
            raw = [corrected_t(r["folds"][best], r["folds"][k], r["n_train"], r["n_test"]) for k in others]
        holm = dict(zip(others, multipletests(raw, method="holm")[1]))
        for k in ALL:
            rows.append(
                [
                    y,
                    "AUC" if binary else "R2",
                    k,
                    means[k],
                    np.std(r["folds"][k], ddof=1),
                    holm.get(k, np.nan),
                    k == best,
                    friedman,
                ]
            )
    t = pd.DataFrame(
        rows, columns=["Outcome", "Metric", "Learner", "Mean", "SD", "p_vs_best_Holm", "Best", "Friedman_p"]
    )
    fp = t.drop_duplicates("Outcome")[["Outcome", "Friedman_p"]]
    t = t.merge(fp.assign(Friedman_q=fdr(fp["Friedman_p"]))[["Outcome", "Friedman_q"]], on="Outcome")
    return t


def main():
    d = load_data()
    cv = cross_validate(d)
    with open(CV_FILE, "wb") as f:
        pickle.dump(cv, f)
    summary(cv).to_excel(RESULTS / "TableS3.xlsx", index=False)


if __name__ == "__main__":
    main()
