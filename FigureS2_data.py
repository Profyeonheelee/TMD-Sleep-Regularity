"""Supplementary Figure S2: fold-level cross-validated performance (requires TableS3.py)."""

import pickle

import pandas as pd

from common import RESULTS, write_sheets
from TableS3 import CV_FILE, K, OUTCOMES, summary


def main():
    with open(CV_FILE, "rb") as f:
        cv = pickle.load(f)
    rows = []
    for y, binary in OUTCOMES:
        for learner, scores in cv[y]["folds"].items():
            for i, s in enumerate(scores):
                rows.append([y, "AUC" if binary else "R2", learner, i // K + 1, i % K + 1, s])
    folds = pd.DataFrame(rows, columns=["Outcome", "Metric", "Learner", "Repeat", "Fold", "Score"])
    table = summary(cv).drop(columns="Metric")
    write_sheets({"FigS2_cv_folds": folds, "FigS2_cv_summary": table})


if __name__ == "__main__":
    main()
