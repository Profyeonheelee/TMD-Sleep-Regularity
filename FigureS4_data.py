"""Supplementary Figure S4: heterogeneity of the association with DI (causal forest and subgroup models)."""

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from common import M2, load_data, fit, write_sheets

MODIFIERS = [
    "Age",
    "Female",
    "log_dur",
    "Bilateral_pain",
    "PSQI_core",
    "Stress",
    "Bruxism",
    "Clenching",
    "STOP_score",
    "Sleep_duration_h",
    "Midsleep_h",
]


def causal_forest(d):
    df = d.dropna(subset=["DI"] + MODIFIERS).reset_index(drop=True)
    X, T, Y = df[MODIFIERS].values, df["Non_regular"].values, df["DI"].values
    W = pd.get_dummies(df["Entry_quarter"], drop_first=True).astype(float).values
    cf = CausalForestDML(
        model_y=HistGradientBoostingRegressor(max_iter=150, min_samples_leaf=40, random_state=0),
        model_t=HistGradientBoostingClassifier(max_iter=150, min_samples_leaf=40, random_state=0),
        discrete_treatment=True,
        n_estimators=1000,
        min_samples_leaf=20,
        cv=5,
        random_state=0,
    )
    cf.fit(Y, T, X=X, W=W)
    cate = df[["Study_ID", "Age", "Female", "Stress", "Clenching", "PSQI_core"]].copy()
    cate.insert(1, "CATE", cf.effect(X))
    ate = pd.DataFrame([[cf.ate(X), *cf.ate_interval(X, alpha=0.05)]], columns=["ATE", "CI_low", "CI_high"])
    drivers = pd.DataFrame({"Feature": MODIFIERS, "Importance": cf.feature_importances_}).sort_values(
        "Importance", ascending=False
    )
    return cate, ate, drivers


def subgroup_models(d):
    med = d["PSQI_core"].median()
    modifiers = [
        ("Age", (d.Age >= 40).astype(float), "Age ≥ 40", "Age < 40"),
        ("Sex", d.Female, "Female", "Male"),
        ("Stress", d.Stress, "Stress", "No stress"),
        ("Clenching", d.Clenching, "Clenching", "No clenching"),
        (
            "Sleep quality",
            (d.PSQI_core >= med).astype(float).where(d.PSQI_core.notna()),
            "PSQI core ≥ median",
            "PSQI core < median",
        ),
    ]
    rows = []
    for y in ["DI", "CMI"]:
        for name, mod, yes, no in modifiers:
            x = d.assign(Mod=mod).dropna(subset=[y, "Age", "Female", "log_dur", "Bilateral_pain", "PSQI_core", "Mod"])
            p_int = fit(f"{y} ~ Non_regular * Mod + {M2}", x, "ols").pvalues["Non_regular:Mod"]
            for level, label in [(1, yes), (0, no)]:
                sub = x[x.Mod == level]
                m = fit(f"{y} ~ Non_regular + {M2}", sub, "ols")
                rows.append(
                    [
                        y,
                        name,
                        label,
                        len(sub),
                        int(sub.Non_regular.sum()),
                        m.params["Non_regular"],
                        *m.conf_int().loc["Non_regular"],
                        m.pvalues["Non_regular"],
                        p_int,
                    ]
                )
    return pd.DataFrame(
        rows,
        columns=["Outcome", "Modifier", "Subgroup", "n", "n_irregular", "B", "CI_low", "CI_high", "p", "p_interaction"],
    )


def main():
    d = load_data()
    cate, ate, drivers = causal_forest(d)
    write_sheets(
        {
            "FigS4_cate_patients": cate,
            "FigS4_ate": ate,
            "FigS4_cate_drivers": drivers,
            "FigS4_interaction": subgroup_models(d),
        }
    )


if __name__ == "__main__":
    main()
