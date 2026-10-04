"""Figure 4: doubly robust estimates across learners, permutation test, E-values and period replication."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import M2, load_data, write_sheets
from gated_transformer import GatedFTTClassifier, GatedFTTRegressor

ADJUST = [
    "Age",
    "Female",
    "log_dur",
    "Bilateral_pain",
    "PSQI_core",
    "STOP_score",
    "Stress",
    "Bruxism",
    "Clenching",
    "Contributing_factors",
    "Systemic_disease",
    "BMI",
]
OUTCOMES = [("DI", False), ("CMI", False), ("Locking", True), ("VAS", False), ("PI", False)]
BASE_LEARNERS = ["Lasso", "GBM", "DNN"]
TRIM = 0.02


# ---------------------------------------------------------------------------
# Double/debiased machine learning (partially linear model) and AIPW
# ---------------------------------------------------------------------------
def design(d, rows, missing_indicators=False):
    q = pd.get_dummies(d["Entry_quarter"], prefix="Q", drop_first=True).astype(float)
    X = pd.concat([d.loc[rows, ADJUST], q.loc[rows]], axis=1).astype(float)
    if missing_indicators:
        for c in ADJUST:
            if d[c].isna().any():
                X["miss_" + c] = X[c].isna().astype(float)
    return X


def learner(name, binary):
    imp = SimpleImputer(strategy="median", add_indicator=True)
    if name == "Lasso":
        est = LogisticRegressionCV(cv=3, Cs=10, max_iter=3000) if binary else LassoCV(cv=3, n_alphas=30, max_iter=5000)
        return make_pipeline(imp, StandardScaler(), est)
    if name == "GBM":
        cls = HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor
        return make_pipeline(
            imp,
            cls(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=0,
            ),
        )
    cls = MLPClassifier if binary else MLPRegressor
    return make_pipeline(
        imp,
        StandardScaler(),
        cls(
            hidden_layer_sizes=(128, 64, 32),
            alpha=1e-2,
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            max_iter=500,
            random_state=0,
        ),
    )


def transformer(binary, seed=0):
    est = GatedFTTClassifier(random_state=seed) if binary else GatedFTTRegressor(random_state=seed)
    return make_pipeline(SimpleImputer(strategy="median"), est)


def predict(model, X, binary):
    return model.predict_proba(X)[:, 1] if binary else model.predict(X)


def estimates(P, T, Y):
    """DML-PLR (residual-on-residual) and AIPW with influence-function standard errors."""
    e = np.clip(P["e"], TRIM, 1 - TRIM)
    rt, ry = T - P["e"], Y - P["l"]
    theta = np.sum(rt * ry) / np.sum(rt * rt)
    psi = (ry - theta * rt) * rt / np.mean(rt * rt)
    se_plr = np.sqrt(np.mean(psi**2) / len(T))
    phi = P["m1"] - P["m0"] + T * (Y - P["m1"]) / e - (1 - T) * (Y - P["m0"]) / (1 - e)
    return theta, se_plr, phi.mean(), phi.std(ddof=1) / np.sqrt(len(T))


def crossfit(X, T, Y, binary, seed):
    P = {k: {q: np.zeros(len(T)) for q in ["e", "l", "m1", "m0"]} for k in BASE_LEARNERS}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, T):
        for k in BASE_LEARNERS:
            P[k]["e"][te] = learner(k, True).fit(X.iloc[tr], T[tr]).predict_proba(X.iloc[te])[:, 1]
            P[k]["l"][te] = predict(learner(k, binary).fit(X.iloc[tr], Y[tr]), X.iloc[te], binary)
        for arm, key in [(1, "m1"), (0, "m0")]:
            idx = tr[T[tr] == arm]
            for k in BASE_LEARNERS:
                P[k][key][te] = predict(learner(k, binary).fit(X.iloc[idx], Y[idx]), X.iloc[te], binary)
    P["Ensemble"] = {q: np.mean([P[k][q] for k in BASE_LEARNERS], axis=0) for q in ["e", "l", "m1", "m0"]}
    return P


def median_aggregate(values):
    """Median estimate over repeated cross-fitting with variance inflated by between-repeat spread."""
    v = np.array(values)
    th = np.median(v[:, 0])
    se = np.sqrt(np.median(v[:, 1] ** 2 + (v[:, 0] - th) ** 2))
    ate = np.median(v[:, 2])
    sea = np.sqrt(np.median(v[:, 3] ** 2 + (v[:, 2] - ate) ** 2))
    return th, se, ate, sea


def result_rows(y, binary, learner_name, th, se, ate, sea, n):
    rows = []
    for estimator, v, s in [("DML-PLR", th, se), ("AIPW", ate, sea)]:
        p = 2 * (1 - stats.norm.cdf(abs(v / s)))
        rows.append(
            [y, learner_name, estimator, "Risk difference" if binary else "B", v, v - 1.96 * s, v + 1.96 * s, p, n]
        )
    return rows


def dml_learners(d, repeats=5):
    rows = []
    for y, binary in OUTCOMES:
        df = d[d[y].notna()]
        X, T, Y = design(d, df.index), df["Non_regular"].values.astype(int), df[y].values.astype(float)
        reps = {}
        for r in range(repeats):
            for k, P in crossfit(X, T, Y, binary, seed=r).items():
                reps.setdefault(k, []).append(estimates(P, T, Y))
        for k, v in reps.items():
            rows += result_rows(y, binary, k, *median_aggregate(v), len(Y))
    return rows


def dml_transformer(d):
    """Single 5-fold cross-fit; propensity model fitted once on the full cohort."""
    T_all = d["Non_regular"].values.astype(int)
    X_all = design(d, d.index, missing_indicators=True)
    e_all = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X_all, T_all):
        e_all[te] = transformer(True).fit(X_all.iloc[tr], T_all[tr]).predict_proba(X_all.iloc[te])[:, 1]
    rows = []
    for y, binary in OUTCOMES[:4]:
        df = d[d[y].notna()]
        X, T, Y = (
            design(d, df.index, missing_indicators=True),
            df["Non_regular"].values.astype(int),
            df[y].values.astype(float),
        )
        P = {"e": e_all[df.index.values], "l": np.zeros(len(Y)), "m1": np.zeros(len(Y)), "m0": np.zeros(len(Y))}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, T):
            P["l"][te] = predict(transformer(binary).fit(X.iloc[tr], Y[tr]), X.iloc[te], binary)
            for arm, key in [(1, "m1"), (0, "m0")]:
                idx = tr[T[tr] == arm]
                P[key][te] = predict(transformer(binary).fit(X.iloc[idx], Y[idx]), X.iloc[te], binary)
        rows += result_rows(y, binary, "Gated Transformer", *estimates(P, T, Y), len(Y))
    return rows


# ---------------------------------------------------------------------------
# Permutation test, E-values and period replication (Model 2)
# ---------------------------------------------------------------------------
def e_value(rr):
    rr = rr if rr >= 1 else 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


def validation(d, n_perm=1000, seed=2026):
    rng = np.random.default_rng(seed)
    perm, ev, rep = [], [], []
    for y, binary in [("DI", False), ("CMI", False), ("Locking", True), ("VAS", False), ("PI", False)]:
        df = d.dropna(subset=[y, "Age", "Female", "log_dur", "Bilateral_pain", "PSQI_core"]).copy()

        def model(f, data):
            return smf.logit(f, data).fit(disp=0) if binary else smf.ols(f, data).fit()

        m = model(f"{y} ~ Non_regular + {M2}", df)
        obs = m.params["Non_regular"]
        null = []
        for _ in range(n_perm):
            df["T_perm"] = df.groupby("Entry_quarter")["Non_regular"].transform(lambda s: rng.permutation(s.values))
            null.append(model(f"{y} ~ T_perm + {M2}", df).params["T_perm"])
        null = np.array(null)
        p_perm = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
        perm += [[y, "logit" if binary else "ols", obs, p_perm, i + 1, v] for i, v in enumerate(null)]

        # E-value: standardized difference -> RR = exp(0.91 d); common binary outcome -> RR = sqrt(OR)
        lo, hi = m.conf_int().loc["Non_regular"]
        to_rr = (lambda b: np.sqrt(np.exp(b))) if binary else (lambda b: np.exp(0.91 * b / df[y].std()))
        rr, rr_ci = to_rr(obs), to_rr(lo if obs > 0 else hi)
        ev.append(
            [
                y,
                np.exp(obs) if binary else obs,
                e_value(rr),
                e_value(rr_ci) if (rr_ci - 1) * (rr - 1) > 0 else 1.0,
                p_perm,
            ]
        )

        for period, sub in [("2021–2022", df[df.Year <= 2022]), ("2023", df[df.Year == 2023])]:
            f = f"{y} ~ Non_regular + {M2}"
            mm = smf.logit(f, sub).fit(disp=0) if binary else smf.ols(f, sub).fit(cov_type="HC3")
            est, a, b = mm.params["Non_regular"], *mm.conf_int().loc["Non_regular"]
            if binary:
                est, a, b = np.exp([est, a, b])
            rep.append([y, period, est, a, b, mm.pvalues["Non_regular"], int(mm.nobs)])
    return (
        pd.DataFrame(perm, columns=["Outcome", "Statistic", "Observed", "Perm_p", "Iteration", "Null"]),
        pd.DataFrame(ev, columns=["Outcome", "Estimate", "E_point", "E_CI", "Perm_p"]),
        pd.DataFrame(rep, columns=["Outcome", "Period", "Estimate", "CI_low", "CI_high", "p", "n"]),
    )


def main():
    d = load_data()
    learners = pd.DataFrame(
        dml_learners(d) + dml_transformer(d),
        columns=["Outcome", "Learner", "Estimator", "Scale", "Effect", "CI_low", "CI_high", "p", "n"],
    )
    perm, ev, rep = validation(d)
    write_sheets({"Fig4a_learners": learners, "Fig4b_permutation": perm, "Fig4c_evalue": ev, "Fig4d_period": rep})


if __name__ == "__main__":
    main()
