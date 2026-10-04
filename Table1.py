"""Table 1 (regular vs irregular sleepers) and Supplementary Table S1 (three schedule subgroups)."""

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import RESULTS, load_data, fdr

ROWS = [
    ("Demographics", None, None),
    ("Age, years", "Age", "mean"),
    ("Female", "Female", "bin"),
    ("BMI, kg/m2", "BMI", "mean"),
    ("Clinical characteristics", None, None),
    ("Symptom duration, months", "Symptom_duration_mo", "median"),
    ("Chronic pain (>=3 months)", "Chronic_3mo", "bin"),
    ("Bilateral pain", "Bilateral_pain", "bin"),
    ("Pain", None, None),
    ("VAS (0-10)", "VAS", "mean"),
    ("PI (0-1)", "PI", "median"),
    ("Jaw function", None, None),
    ("DI (0-1)", "DI", "median"),
    ("CMI (0-1)", "CMI", "median"),
    ("Locking", "Locking", "bin"),
    ("Joint noise", "Joint_noise", "bin"),
    ("MUO, mm", "MUO_mm", "mean"),
    ("Sleep", None, None),
    ("Bedtime, clock time", "Bedtime_clock", "clock"),
    ("Wake time, clock time", "Wake_time_h", "clock"),
    ("Mid-sleep, clock time", "Midsleep_h", "clock"),
    ("Sleep duration, h", "Sleep_duration_h", "mean"),
    ("Short sleep (below NSF range)", "Short_sleep", "bin"),
    ("PSQI core score (0-15)", "PSQI_core", "mean"),
    ("PSQI global score (0-21)", "PSQI_global", "mean"),
    ("Poor sleepers (PSQI > 5)", "Poor_sleeper", "bin"),
    ("STOP score (0-4)", "STOP_score", "mean"),
    ("Behavioral and psychological factors", None, None),
    ("Bruxism", "Bruxism", "bin"),
    ("Clenching", "Clenching", "bin"),
    ("Stress", "Stress", "bin"),
    ("Contributing factors, n", "Contributing_factors", "median"),
    ("Systemic disease", "Systemic_disease", "bin"),
]
SUBGROUPS = ["Regular", "Variable timing", "No fixed schedule"]


def hhmm(h):
    h = h % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def summarise(x, kind):
    x = x.dropna()
    if kind == "bin":
        return f"{int(x.sum()):,} ({x.mean() * 100:.1f})"
    if kind == "median":
        f = ".2f" if x.max() <= 1.5 else ".1f"
        return f"{x.median():{f}} [{x.quantile(.25):{f}}-{x.quantile(.75):{f}}]"
    if kind == "clock":
        return f"{hhmm(x.mean())} ± {x.std() * 60:.0f} min"
    f = ".1f" if x.mean() > 2 else ".2f"
    return f"{x.mean():{f}} ± {x.std():{f}}"


def smd(a, b, kind):
    a, b = a.dropna(), b.dropna()
    if kind == "bin":
        p1, p2 = a.mean(), b.mean()
        s = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        return (p1 - p2) / s if s > 0 else np.nan
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


def two_group_test(a, b, kind):
    a, b = a.dropna(), b.dropna()
    if kind == "bin":
        ct = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
        if ct.min() < 5:
            return stats.fisher_exact(ct)[1], "Fisher"
        return stats.chi2_contingency(ct)[1], "Chi-square"
    if kind == "median":
        return stats.mannwhitneyu(a, b).pvalue, "Mann-Whitney U"
    return stats.ttest_ind(a, b, equal_var=False).pvalue, "Welch t"


def table1(d):
    reg, irr = d[d.Group == "Regular"], d[d.Group == "Irregular"]
    rows = []
    for label, var, kind in ROWS:
        if var is None:
            rows.append([label, "", "", "", np.nan, "", np.nan])
            continue
        p, test = two_group_test(reg[var], irr[var], kind)
        rows.append(
            [
                label,
                summarise(d[var], kind),
                summarise(reg[var], kind),
                summarise(irr[var], kind),
                p,
                test,
                smd(irr[var], reg[var], kind),
            ]
        )
    t = pd.DataFrame(
        rows,
        columns=[
            "Characteristic",
            f"Total (n = {len(d):,})",
            f"Regular sleepers (n = {len(reg):,})",
            f"Irregular sleepers (n = {len(irr):,})",
            "p",
            "Test",
            "SMD",
        ],
    )
    m = t["p"].notna()
    t.loc[m, "q"] = fdr(t.loc[m, "p"])
    return t


def table_s1(d):
    rows = []
    for label, var, kind in ROWS:
        if var is None:
            rows.append([label, "", "", "", np.nan, "", np.nan, np.nan, np.nan])
            continue
        x = d[["Subgroup", var]].dropna()
        groups = [x[x.Subgroup == g][var] for g in SUBGROUPS]
        cells = [summarise(g, kind) if len(g) > 1 else f"n = {len(g)}" for g in groups]
        if kind == "bin":
            ct = pd.crosstab(x.Subgroup, x[var]).reindex(SUBGROUPS)
            p, test = stats.chi2_contingency(ct)[1], "Chi-square"
            raw = []
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                sub = ct.iloc[[i, j]].values
                raw.append(stats.fisher_exact(sub)[1] if sub.min() < 5 else stats.chi2_contingency(sub)[1])
            post = multipletests(raw, method="holm")[1]
        elif kind == "median":
            p, test = stats.kruskal(*groups).pvalue, "Kruskal-Wallis"
            dunn = sp.posthoc_dunn(x, val_col=var, group_col="Subgroup", p_adjust="holm")
            post = [
                dunn.loc[SUBGROUPS[0], SUBGROUPS[1]],
                dunn.loc[SUBGROUPS[0], SUBGROUPS[2]],
                dunn.loc[SUBGROUPS[1], SUBGROUPS[2]],
            ]
        else:
            if min(len(g) for g in groups) < 2:
                rows.append([label, *cells, np.nan, "Not tested", np.nan, np.nan, np.nan])
                continue
            p, test = stats.f_oneway(*groups).pvalue, "ANOVA"
            tk = stats.tukey_hsd(*groups).pvalue
            post = [tk[0, 1], tk[0, 2], tk[1, 2]]
        rows.append([label, *cells, p, test, *post])
    n = d.Subgroup.value_counts()
    t = pd.DataFrame(
        rows,
        columns=["Characteristic"]
        + [f"{g} (n = {n[g]:,})" for g in SUBGROUPS]
        + ["p", "Test", "Regular vs Variable", "Regular vs No fixed", "Variable vs No fixed"],
    )
    m = t["p"].notna()
    t.insert(t.columns.get_loc("Test"), "q", np.nan)
    t.loc[m, "q"] = fdr(t.loc[m, "p"])
    return t


def main():
    d = load_data()
    d["Bedtime_clock"] = (24 - d["Bedtime_h"]) % 24
    d["Bedtime_clock"] = np.where(d["Bedtime_clock"] < 12, d["Bedtime_clock"] + 24, d["Bedtime_clock"])
    t1, s1 = table1(d), table_s1(d)
    with pd.ExcelWriter(RESULTS / "Table1.xlsx") as w:
        t1.to_excel(w, sheet_name="Table1", index=False)
        s1.to_excel(w, sheet_name="TableS1", index=False)
    print(t1.round(4).to_string())


if __name__ == "__main__":
    main()
