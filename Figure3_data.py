"""Figure 3: sequential adjustment (with bootstrap), oral behaviour prevalence and mediation."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import M1, M2, M3, M4, M5, load_data, model4_sample, fit, write_sheets

STEPS = [
    ("Base model", "Age, sex, symptom duration, bilateral pain, entry quarter", M1),
    ("+ Sleep quality", "PSQI core score", M2),
    ("+ Sleep timing and duration", "Mid-sleep timing; NSF duration category", M3),
    ("+ Oral behaviours, stress, OSA risk", "Bruxism, clenching, stress, STOP score, contributing factors", M4),
    ("+ Free-text sleep bruxism and pain-disrupted sleep", "Clinician notes (rule-based NLP)", M5),
]
OUTCOMES = [("DI", "ols"), ("CMI", "ols"), ("Locking", "logit"), ("VAS", "ols")]
MEDIATORS = [
    ("Clenching", "Clenching"),
    ("Bruxism", "Bruxism"),
    ("Sleep_bruxism_text", "Sleep bruxism (clinician notes)"),
]
N_BOOT_STEPS, N_BOOT_MEDIATION = 500, 1000


def adjustment(base):
    rows = []
    for y, kind in OUTCOMES:
        data = base.dropna(subset=[y])
        ref = None
        for i, (step, added, rhs) in enumerate(STEPS, 1):
            m = fit(f"{y} ~ Non_regular + {rhs}", data, kind)
            b = m.params["Non_regular"]
            lo, hi = m.conf_int().loc["Non_regular"]
            ref = b if ref is None else ref
            est = np.exp([b, lo, hi]) if kind == "logit" else (b, lo, hi)
            rows.append(
                [
                    y,
                    "OR" if kind == "logit" else "B",
                    i,
                    step,
                    added,
                    *est,
                    m.pvalues["Non_regular"],
                    int(m.nobs),
                    (b - ref) / ref * 100,
                ]
            )
    return pd.DataFrame(
        rows,
        columns=[
            "Outcome",
            "Scale",
            "Step",
            "Adjustment",
            "Covariates_added",
            "Estimate",
            "CI_low",
            "CI_high",
            "p",
            "n",
            "Pct_change_vs_base",
        ],
    )


def adjustment_bootstrap(base, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for y, kind in OUTCOMES:
        data = base.dropna(subset=[y]).reset_index(drop=True)
        draws = [rng.integers(0, len(data), len(data)) for _ in range(N_BOOT_STEPS)]
        for i, (step, _, rhs) in enumerate(STEPS, 1):
            f = f"{y} ~ Non_regular + {rhs}"
            for b, idx in enumerate(draws, 1):
                sample = data.iloc[idx]
                try:
                    m = smf.ols(f, sample).fit() if kind == "ols" else smf.logit(f, sample).fit(disp=0, maxiter=100)
                    v = m.params["Non_regular"]
                    v = np.exp(v) if kind == "logit" else v
                except Exception:
                    v = np.nan
                rows.append([y, i, step, b, v])
    return pd.DataFrame(rows, columns=["Outcome", "Step", "Adjustment", "Boot", "Estimate"])


def prevalence(d):
    rows = []
    for var, label in MEDIATORS:
        x = d[["Non_regular", var]].dropna()
        p = stats.chi2_contingency(pd.crosstab(x.Non_regular, x[var]))[1]
        for g, name in [(0, "Regular sleepers"), (1, "Irregular sleepers")]:
            s = x.loc[x.Non_regular == g, var]
            rows.append([label, name, int(s.sum()), len(s), s.mean() * 100, p])
    return pd.DataFrame(rows, columns=["Behaviour", "Group", "n_yes", "n", "Pct", "p_chi2"])


def paths(data, y, mediator):
    a = smf.ols(f"{mediator} ~ Non_regular + {M2}", data).fit().params["Non_regular"]
    b = smf.ols(f"{y} ~ Non_regular + {mediator} + {M2}", data).fit().params[mediator]
    return a, b, a * b


def mediation(base, seed=1):
    """Product-of-coefficients mediation with Model 2 covariates; percentile bootstrap CIs."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in ["DI", "CMI"]:
        data = base.dropna(subset=[y]).reset_index(drop=True)
        total = smf.ols(f"{y} ~ Non_regular + {M2}", data).fit()
        t = total.params["Non_regular"]
        for var, label in MEDIATORS:
            a, b, ab = paths(data, y, var)
            boot = np.array(
                [paths(data.iloc[rng.integers(0, len(data), len(data))], y, var) for _ in range(N_BOOT_MEDIATION)]
            )
            ci = lambda k: np.percentile(boot[:, k], [2.5, 97.5])
            rows.append(
                [
                    y,
                    label,
                    a,
                    *ci(0),
                    b,
                    *ci(1),
                    ab,
                    *ci(2),
                    t,
                    *total.conf_int().loc["Non_regular"],
                    ab / t * 100,
                    len(data),
                ]
            )
    return pd.DataFrame(
        rows,
        columns=[
            "Outcome",
            "Mediator",
            "a",
            "a_low",
            "a_high",
            "b",
            "b_low",
            "b_high",
            "Indirect",
            "Ind_low",
            "Ind_high",
            "Total",
            "Tot_low",
            "Tot_high",
            "Pct_mediated",
            "n",
        ],
    )


def mediation_bootstrap(base, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for y in ["DI", "CMI"]:
        data = base.dropna(subset=[y]).reset_index(drop=True)
        for b in range(1, N_BOOT_MEDIATION + 1):
            sample = data.iloc[rng.integers(0, len(data), len(data))]
            rows.append(
                [y, "Total effect", b, smf.ols(f"{y} ~ Non_regular + {M2}", sample).fit().params["Non_regular"]]
            )
            for var, label in [
                ("Clenching", "via clenching"),
                ("Bruxism", "via bruxism"),
                ("Sleep_bruxism_text", "via sleep bruxism"),
            ]:
                rows.append([y, label, b, paths(sample, y, var)[2]])
    return pd.DataFrame(rows, columns=["Outcome", "Path", "Boot", "Effect"])


def main():
    d = load_data(fill_undetermined=True)
    base = model4_sample(d)
    write_sheets(
        {
            "Fig3a_adjustment": adjustment(base),
            "Fig3a_bootstrap": adjustment_bootstrap(base),
            "Fig3b_prevalence": prevalence(d),
            "Fig3c_mediation": mediation(base),
            "Fig3c_mediation_boot": mediation_bootstrap(base),
        }
    )


if __name__ == "__main__":
    main()
