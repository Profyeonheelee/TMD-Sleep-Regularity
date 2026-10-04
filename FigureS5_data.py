"""Supplementary Figure S5: regularity, timing and duration modeled together (Model 2 + timing + duration)."""

import pandas as pd

from common import M2, load_data, model4_sample, fit, coef, fdr, write_sheets

RHS = M2 + " + C(Sleep_timing, Treatment('Night-aligned')) + C(Sleep_duration_cat, Treatment('Recommended'))"
TERMS = {
    "Non_regular": "Irregular sleep",
    "C(Sleep_timing, Treatment('Night-aligned'))[T.Advanced]": "Advanced timing",
    "C(Sleep_timing, Treatment('Night-aligned'))[T.Delayed]": "Delayed timing",
    "C(Sleep_duration_cat, Treatment('Recommended'))[T.Short]": "Short sleep",
    "C(Sleep_duration_cat, Treatment('Recommended'))[T.Long]": "Long sleep",
}
OUTCOMES = [("VAS", "ols"), ("PI", "ols"), ("DI", "ols"), ("CMI", "ols"), ("Locking", "logit")]


def main():
    base = model4_sample(load_data(fill_undetermined=True))
    rows = []
    for y, kind in OUTCOMES:
        m = fit(f"{y} ~ Non_regular + {RHS}", base.dropna(subset=[y]), kind)
        for term, label in TERMS.items():
            rows.append([y, label, "OR" if kind == "logit" else "B", *coef(m, term, kind), int(m.nobs)])
    out = pd.DataFrame(rows, columns=["Outcome", "Exposure", "Scale", "Estimate", "CI_low", "CI_high", "p", "n"])
    out["q"] = out.groupby("Outcome")["p"].transform(lambda p: fdr(p))
    write_sheets({"FigS5_timing_duration": out})


if __name__ == "__main__":
    main()
