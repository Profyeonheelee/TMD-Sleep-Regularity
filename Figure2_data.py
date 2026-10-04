"""Figure 2: standardized estimates across Models 1-4 and equivalence tests (TOST) for pain."""

import numpy as np
import pandas as pd
from scipy import stats

from common import MODELS, load_data, model4_sample, fit, write_sheets
from Table2 import table2

LOGIT_TO_SD = np.sqrt(3) / np.pi


def standardized(t2, d):
    sd = {y: d[y].std() for y in ["VAS", "PI", "DI", "CMI"]}
    t = t2.copy()
    for col, std in [("Estimate", "Std_est"), ("CI_low", "Std_low"), ("CI_high", "Std_high")]:
        t[std] = [v / sd[y] if s == "B" else np.log(v) * LOGIT_TO_SD for v, y, s in zip(t[col], t.Outcome, t.Scale)]
    t["Outcome"] = t["Outcome"].replace({"Joint_noise": "Joint noise"})
    return t.rename(columns={"q": "q_FDR"})[
        [
            "Model",
            "Outcome",
            "Domain",
            "Scale",
            "Estimate",
            "CI_low",
            "CI_high",
            "p",
            "q_FDR",
            "n",
            "Std_est",
            "Std_low",
            "Std_high",
        ]
    ]


def equivalence(base):
    """Two one-sided tests: equivalence if the 90% CI lies within +/- margin."""
    rows = []
    for y in ["VAS", "PI"]:
        data = base.dropna(subset=[y])
        sd = data[y].std()
        margins = [("MCID 1 point", 1.0), ("0.2 SD", 0.2 * sd)] if y == "VAS" else [("0.2 SD", 0.2 * sd)]
        for model, rhs in MODELS:
            m = fit(f"{y} ~ Non_regular + {rhs}", data, "ols")
            b, se = m.params["Non_regular"], m.bse["Non_regular"]
            lo, hi = b - 1.645 * se, b + 1.645 * se
            for name, margin in margins:
                p = max(stats.norm.sf((b + margin) / se), stats.norm.cdf((b - margin) / se))
                rows.append([y, model, b, lo, hi, name, margin, p, int(m.nobs), sd])
    return pd.DataFrame(
        rows, columns=["Outcome", "Model", "B", "CI90_low", "CI90_high", "Margin_type", "Margin", "p_TOST", "n", "SD"]
    )


def main():
    d = load_data(fill_undetermined=True)
    base = model4_sample(d)
    write_sheets({"Fig2a_models": standardized(table2(base), d), "Fig2b_equivalence": equivalence(base)})


if __name__ == "__main__":
    main()
