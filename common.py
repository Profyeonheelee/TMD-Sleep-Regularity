"""Shared paths, variable derivation, model formulas and helpers."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ANALYSIS_FILE = DATA_DIR / "analysis_dataset.xlsx"
FIGURE_DATA = RESULTS / "Figure_data.xlsx"

# Model 1-4 covariate sets (Methods, Covariates)
M1 = "Age + Female + log_dur + Bilateral_pain + C(Entry_quarter)"
M2 = M1 + " + PSQI_core"
M3 = M2 + " + C(Sleep_timing, Treatment('Night-aligned')) + C(Sleep_duration_cat, Treatment('Recommended'))"
M4 = M3 + " + Bruxism + Clenching + Stress + STOP_score + Contributing_factors"
M5 = M4 + " + Sleep_bruxism_text + Pain_sleep_text"
MODELS = [("Model 1", M1), ("Model 2", M2), ("Model 3", M3), ("Model 4", M4)]
MODEL4_VARS = [
    "Age",
    "Female",
    "log_dur",
    "Bilateral_pain",
    "PSQI_core",
    "Sleep_timing",
    "Sleep_duration_cat",
    "Bruxism",
    "Clenching",
    "Stress",
    "STOP_score",
    "Contributing_factors",
]

OUTCOMES = [
    ("Pain", "VAS", "ols"),
    ("Pain", "PI", "ols"),
    ("Jaw function", "DI", "ols"),
    ("Jaw function", "CMI", "ols"),
    ("Jaw function", "Locking", "logit"),
    ("Jaw function", "Joint_noise", "logit"),
]


def load_data(fill_undetermined=False):
    """Analysis dataset with derived sleep timing and duration categories."""
    d = pd.read_excel(ANALYSIS_FILE, sheet_name="Data")
    d["Entry_quarter"] = d["Entry_quarter"].astype(str)
    d["Year"] = d["Entry_quarter"].str[:4].astype(int)
    d["Entry_period"] = d["Entry_quarter"].map({q: i for i, q in enumerate(sorted(d["Entry_quarter"].unique()))})
    d["log_dur"] = np.log1p(d["Symptom_duration_mo"])
    d["Group"] = np.where(d["Non_regular"] == 1, "Irregular", "Regular")
    d["Subgroup"] = d["Sleep_schedule"].map(
        {"Regular": "Regular", "Variable": "Variable timing", "Irregular": "No fixed schedule"}
    )

    # mid-sleep (clock hour); Bedtime_h is hours before midnight, Wake_time_h is clock hour
    ms = ((d["Wake_time_h"] - d["Bedtime_h"]) / 2) % 24
    d["Midsleep_h"] = ms
    d["Sleep_timing"] = pd.Series(
        np.select([ms < 2, ms < 5], ["Advanced", "Night-aligned"], "Delayed"), index=d.index
    ).where(ms.notna())
    d["Delayed_timing"] = (ms >= 5).astype(float).where(ms.notna())

    # National Sleep Foundation age-specific recommended duration (Hirshkowitz et al., 2015)
    age, hrs = d["Age"], d["Sleep_duration_h"]
    lo = np.select([age < 14, age < 18, age < 65], [9, 8, 7], 7)
    hi = np.select([age < 14, age < 18, age < 65], [11, 10, 9], 8)
    d["Sleep_duration_cat"] = pd.Series(
        np.select([hrs < lo, hrs > hi], ["Short", "Long"], "Recommended"), index=d.index
    ).where(hrs.notna())
    d["Short_sleep"] = (hrs < lo).astype(float).where(hrs.notna())
    d["Long_sleep"] = (hrs > hi).astype(float).where(hrs.notna())

    d["Chronic_3mo"] = (d["Symptom_duration_mo"] >= 3).astype(float).where(d["Symptom_duration_mo"].notna())

    if fill_undetermined:
        d["Sleep_timing"] = d["Sleep_timing"].fillna("Undetermined")
        d["Sleep_duration_cat"] = d["Sleep_duration_cat"].fillna("Undetermined")
    return d


def model4_sample(d):
    """Common complete-case sample for Models 1-4."""
    return d.dropna(subset=MODEL4_VARS).copy()


def fit(formula, data, kind):
    if kind == "ols":
        return smf.ols(formula, data).fit(cov_type="HC3")
    return smf.logit(formula, data).fit(disp=0, maxiter=200)


def coef(model, term, kind):
    """Estimate, 95% CI and p value; exponentiated for logistic models."""
    est, lo, hi = model.params[term], *model.conf_int().loc[term]
    if kind == "logit":
        est, lo, hi = np.exp([est, lo, hi])
    return est, lo, hi, model.pvalues[term]


def fdr(p):
    return multipletests(p, method="fdr_bh")[1]


def write_sheets(sheets, path=FIGURE_DATA):
    """Add or replace sheets in the figure data workbook."""
    mode = "a" if Path(path).exists() else "w"
    kw = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(path, mode=mode, engine="openpyxl", **kw) as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
