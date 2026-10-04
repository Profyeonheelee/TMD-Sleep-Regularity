"""Table 2 (irregular vs regular sleepers, Models 1-4) and Supplementary Table S2 (schedule subgroups)."""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common import RESULTS, MODELS, M2, M4, OUTCOMES, load_data, model4_sample, fit, coef, fdr

GROUP = "C(Group, Treatment('Regular'))"
SUBGROUP = "C(Subgroup3, Treatment('Regular'))"


def table2(base):
    rows = []
    for model, rhs in MODELS:
        for domain, y, kind in OUTCOMES:
            data = base.dropna(subset=[y])
            m = fit(f"{y} ~ {GROUP} + {rhs}", data, kind)
            est, lo, hi, p = coef(m, f"{GROUP}[T.Irregular]", kind)
            rows.append([model, domain, y, "OR" if kind == "logit" else "B", est, lo, hi, p, int(m.nobs)])
    t = pd.DataFrame(rows, columns=["Model", "Domain", "Outcome", "Scale", "Estimate", "CI_low", "CI_high", "p", "n"])
    t["q"] = t.groupby("Model")["p"].transform(lambda p: fdr(p))
    return t


def group_values(base):
    rows = []
    for _, y, kind in OUTCOMES:
        data = base.dropna(subset=[y])
        for g in ["Regular", "Irregular"]:
            x = data.loc[data.Group == g, y]
            rows.append(
                [
                    y,
                    g,
                    (
                        f"{int(x.sum()):,} ({x.mean() * 100:.1f})"
                        if kind == "logit"
                        else f"{x.mean():.2f} ± {x.std():.2f}"
                    ),
                ]
            )
    return (
        pd.DataFrame(rows, columns=["Outcome", "Group", "Value"])
        .pivot(index="Outcome", columns="Group", values="Value")
        .reset_index()
    )


def table_s2(base):
    base = base.copy()
    base["Subgroup3"] = base["Sleep_schedule"].map(
        {"Regular": "Regular", "Variable": "Variable", "Irregular": "NoFixed"}
    )
    rows = []
    for model, rhs in [("Model 2", M2), ("Model 4", M4)]:
        for _, y, kind in OUTCOMES:
            data = base.dropna(subset=[y])
            m = fit(f"{y} ~ {SUBGROUP} + {rhs}", data, kind)
            kv, kn = f"{SUBGROUP}[T.Variable]", f"{SUBGROUP}[T.NoFixed]"
            contrasts = [
                ("Variable timing vs Regular", m.params[kv], *m.conf_int().loc[kv], m.pvalues[kv]),
                ("No fixed schedule vs Regular", m.params[kn], *m.conf_int().loc[kn], m.pvalues[kn]),
            ]
            t = m.t_test(f"{kn} - {kv} = 0")
            ci = np.asarray(t.conf_int()).ravel()
            contrasts.append(
                (
                    "No fixed schedule vs Variable timing",
                    float(np.squeeze(t.effect)),
                    ci[0],
                    ci[1],
                    float(np.squeeze(t.pvalue)),
                )
            )
            holm = multipletests([c[4] for c in contrasts], method="holm")[1]
            for (name, est, lo, hi, p), ph in zip(contrasts, holm):
                if kind == "logit":
                    est, lo, hi = np.exp([est, lo, hi])
                rows.append([model, y, name, "OR" if kind == "logit" else "B", est, lo, hi, p, ph])
    return pd.DataFrame(
        rows, columns=["Model", "Outcome", "Contrast", "Scale", "Estimate", "CI_low", "CI_high", "p", "p_Holm"]
    )


def main():
    base = model4_sample(load_data(fill_undetermined=True))
    t2, s2 = table2(base), table_s2(base)
    with pd.ExcelWriter(RESULTS / "Table2.xlsx") as w:
        t2.to_excel(w, sheet_name="Table2", index=False)
        group_values(base).to_excel(w, sheet_name="Table2_group_values", index=False)
        s2.to_excel(w, sheet_name="TableS2", index=False)
    print(f"n = {len(base)}; {base.Group.value_counts().to_dict()}")
    print(t2.round(4).to_string())


if __name__ == "__main__":
    main()
