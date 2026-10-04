"""Figure 5: SHAP feature contributions of the gated attention Transformer for each outcome."""

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import load_data, write_sheets
from gated_transformer import GatedFTTClassifier, GatedFTTRegressor
from TableS3 import features

LABELS = {
    "Non_regular": ("Irregular schedule", "Regularity"),
    "Schedule_range_h": ("Schedule range (h)", "Regularity"),
    "Midsleep_h": ("Mid-sleep time", "Timing"),
    "Bedtime_h": ("Bedtime", "Timing"),
    "Sleep_duration_h": ("Sleep duration (h)", "Duration"),
    "PSQI_C1": ("Sleep quality (C1)", "Satisfaction"),
    "PSQI_C2": ("Sleep latency (C2)", "Efficiency"),
    "PSQI_C4": ("Sleep efficiency (C4)", "Efficiency"),
    "PSQI_C5": ("Sleep disturbance (C5)", "Satisfaction"),
    "PSQI_C6": ("Sleep medication (C6)", "Satisfaction"),
    "PSQI_C7": ("Daytime dysfunction (C7)", "Alertness"),
    "STOP_score": ("STOP score", "Sleep-disordered breathing"),
    "Age": ("Age", "Clinical"),
    "Female": ("Female sex", "Clinical"),
    "BMI": ("BMI", "Clinical"),
    "log_dur": ("Symptom duration", "Clinical"),
    "Bilateral_pain": ("Bilateral pain", "Clinical"),
    "Stress": ("Stress", "Behavioral"),
    "Bruxism": ("Bruxism", "Behavioral"),
    "Clenching": ("Clenching", "Behavioral"),
    "Contributing_factors": ("Contributing factors", "Behavioral"),
    "Systemic_disease": ("Systemic disease", "Clinical"),
    "Entry_period": ("Entry period", "Design"),
}
OUTCOMES = [("DI", False), ("CMI", False), ("Locking", True), ("VAS", False)]


def label(f):
    if f in LABELS:
        return LABELS[f]
    return (f, "Missing indicator") if f.startswith("miss_") else (f, "Other")


class _Output(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).unsqueeze(-1)


def shap_values(X, Y, binary, n_explain=600, n_background=200):
    model = make_pipeline(
        SimpleImputer(strategy="median"), GatedFTTClassifier() if binary else GatedFTTRegressor()
    ).fit(X, Y)
    Z = model[0].transform(X).astype(np.float32)
    est = model[-1]
    Zs = torch.tensor((Z - est.mu_) / est.sd_)
    idx = np.random.default_rng(2).choice(len(Zs), n_explain, replace=False)
    background = Zs[np.random.default_rng(3).choice(len(Zs), n_background, replace=False)]
    est.net_.eval()
    sv = shap.GradientExplainer(_Output(est.net_), background).shap_values(Zs[idx])
    return np.array(sv).reshape(len(idx), -1), Z[idx]


def main():
    d = load_data()
    importance, beeswarm = [], []
    for y, binary in OUTCOMES:
        df = d[d[y].notna()].reset_index(drop=True)
        X = features(df)
        sv, xv = shap_values(X, df[y].values.astype(float), binary)
        for j, f in enumerate(X.columns):
            importance.append([y, f, *label(f), np.abs(sv[:, j]).mean(), "Mean |SHAP|"])
            if y in ("DI", "VAS") and not f.startswith("miss_"):
                x = xv[:, j]
                scaled = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)
                beeswarm += [[y, *label(f), sv[i, j], x[i], scaled[i]] for i in range(len(x))]
        print(y, "done", flush=True)
    write_sheets(
        {
            "Fig5a_importance": pd.DataFrame(
                importance, columns=["Outcome", "Feature", "Label", "Domain", "Value", "Metric"]
            ),
            "Fig5b_shap": pd.DataFrame(
                beeswarm, columns=["Outcome", "Label", "Domain", "SHAP", "Feature_value", "Feature_value_scaled"]
            ),
        }
    )


if __name__ == "__main__":
    main()
