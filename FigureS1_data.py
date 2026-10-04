"""Supplementary Figure S1: irregular sleep defined by a range of >=1 h (main) or >=2 h (sensitivity)."""

import numpy as np
import pandas as pd

from common import M2, load_data, fit, coef, write_sheets

OUTCOMES = [("DI", "ols"), ("CMI", "ols"), ("Locking", "logit"), ("VAS", "ols")]


def main():
    d = load_data()
    d["Non_regular_2h"] = np.where(d["Sleep_schedule"] == "Irregular", 1, (d["Schedule_range_h"] >= 2).astype(float))
    d.loc[d["Schedule_range_h"].isna() & (d["Sleep_schedule"] != "Irregular"), "Non_regular_2h"] = np.nan
    rows = []
    for y, kind in OUTCOMES:
        for analysis, x in [
            ("Main (range ≥1 h, Model 2)", "Non_regular"),
            ("Sensitivity: range ≥2 h", "Non_regular_2h"),
        ]:
            m = fit(f"{y} ~ {x} + {M2}", d, kind)
            rows.append([y, analysis, "OR" if kind == "logit" else "B", *coef(m, x, kind), int(m.nobs)])
    out = pd.DataFrame(rows, columns=["Outcome", "Analysis", "Effect", "Estimate", "CI_low", "CI_high", "p", "n"])
    write_sheets({"FigS1_threshold": out})


if __name__ == "__main__":
    main()
