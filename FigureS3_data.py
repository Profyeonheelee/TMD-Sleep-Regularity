"""Supplementary Figure S3: ROC curves for locking from out-of-fold predictions (requires TableS3.py)."""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from statsmodels.stats.multitest import multipletests

from common import write_sheets
from TableS3 import ALL, CV_FILE, LEARNERS, delong

GRID = np.linspace(0, 1, 101)


def delong_ci(y, p):
    pos, neg = p[y == 1], p[y == 0]
    v10 = np.array([(np.sum(x > neg) + 0.5 * np.sum(x == neg)) / len(neg) for x in pos])
    v01 = np.array([(np.sum(pos > x) + 0.5 * np.sum(pos == x)) / len(pos) for x in neg])
    auc = v10.mean()
    se = np.sqrt(v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01))
    return auc, auc - 1.96 * se, auc + 1.96 * se


def roc_band(y, p, n_boot=1000, seed=0):
    """Pointwise 95% band of TPR from stratified bootstrap resamples."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    tpr = []
    for _ in range(n_boot):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        f, t, _ = roc_curve(y[i], p[i])
        tpr.append(np.interp(GRID, f, t))
    return np.percentile(tpr, 2.5, axis=0), np.percentile(tpr, 97.5, axis=0)


def main():
    with open(CV_FILE, "rb") as f:
        cv = pickle.load(f)["Locking"]
    y = cv["Y"].astype(int)
    pred = {k: cv["oof"][k][0] for k in ALL}  # first cross-validation repeat

    curves, bands, auc = [], [], []
    p_holm = dict(
        zip(LEARNERS, multipletests([delong(y, pred["Ensemble"], pred[k]) for k in LEARNERS], method="holm")[1])
    )
    for k in ALL:
        fpr, tpr, _ = roc_curve(y, pred[k])
        curves.append(pd.DataFrame({"Learner": k, "FPR": fpr, "TPR": tpr}))
        lo, hi = roc_band(y, pred[k])
        bands.append(pd.DataFrame({"Learner": k, "FPR": GRID, "TPR_low": lo, "TPR_high": hi}))
        auc.append([k, *delong_ci(y, pred[k]), p_holm.get(k, np.nan), len(y), int(y.sum())])
    write_sheets(
        {
            "FigS3_roc": pd.concat(curves, ignore_index=True),
            "FigS3_auc": pd.DataFrame(
                auc, columns=["Learner", "AUC", "CI_low", "CI_high", "p_vs_Ensemble_Holm", "n", "events"]
            ),
            "FigS3_roc_band": pd.concat(bands, ignore_index=True),
        }
    )


if __name__ == "__main__":
    main()
