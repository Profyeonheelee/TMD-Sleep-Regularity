"""Build the analysis dataset.

Inputs (data/):
  TMD_Sleep_Cleaned_Dataset.xlsx  de-identified first-visit records (sheet "Data")
  psqi_freetext.csv               Study_ID and verbatim PSQI answers
                                  (Q1_bedtime, Q2_latency, Q3_waketime, Q4_sleep_hours,
                                   Q5j_other_reason, Q10e_other_problem)
Output:
  data/analysis_dataset.xlsx
"""

import re

import numpy as np
import pandas as pd

from common import DATA_DIR, ANALYSIS_FILE

IRREGULAR_TERMS = (
    "irregular",
    "불규칙",
    "일정하지",
    "일정치",
    "그때그때",
    "들쭉",
    "들쑥",
    "대중없",
    "졸릴",
    "아무때",
    "낮밤",
    "밤낮",
    "때마다",
    "상관없",
)
UNKNOWN_TERMS = ("unknown", "모름", "모르")

# keyword rules applied to the free-text PSQI answers
TEXT_RULES = {
    "Sleep_bruxism_text": r"이갈|이를 ?갈|이 ?갈|악물|깨물|앙다|이를 ?꽉|씹|오물",
    "Pain_sleep_text": r"통증|아파|아픔|아프|저림|두통|턱",
}


def range_width(text, clock=True):
    """Width (h) of a reported range such as '23~1'; 0 for a single time; NaN if not determinable."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan
    u = str(text).replace(" ", "").lower()
    if u in ("", "-", ".") or any(w in u for w in UNKNOWN_TERMS) or any(w in u for w in IRREGULAR_TERMS):
        return np.nan
    hm = re.findall(r"(\d{1,2}):(\d{2})", u)
    vals = [int(a) + int(b) / 60 for a, b in hm] if hm else [float(x) for x in re.findall(r"\d+(?:\.\d+)?", u)]
    if not vals:
        return np.nan
    if len(vals) >= 2 and re.search(r"[~\-,/]", u):
        a, b = vals[0], vals[1]
        if clock and b < a:
            b += 24 if a >= 12 else 12
        return min(abs(b - a), 12)
    return 0.0


def has_irregular_term(text):
    return isinstance(text, str) and any(w in text.replace(" ", "").lower() for w in IRREGULAR_TERMS)


def classify_schedule(txt):
    bed = txt["Q1_bedtime"].apply(range_width)
    wake = txt["Q3_waketime"].apply(range_width)
    irregular = (
        txt["Q1_bedtime"].apply(has_irregular_term)
        | txt["Q3_waketime"].apply(has_irregular_term)
        | txt["Q4_sleep_hours"].apply(has_irregular_term)
    )
    rng = pd.concat([bed, wake], axis=1).max(axis=1, skipna=False)
    sched = pd.Series(np.where(rng >= 1, "Variable", "Regular"), index=txt.index).where(rng.notna())
    sched[irregular] = "Irregular"
    return pd.DataFrame({"Study_ID": txt["Study_ID"], "Sleep_schedule": sched, "Schedule_range_h": rng})


def text_flags(txt):
    def is_time(v):
        return bool(re.fullmatch(r"\s*[\d.:]+\s*", v)) or v == ""

    def joined(r):
        parts = []
        for col in ["Q1_bedtime", "Q3_waketime", "Q4_sleep_hours", "Q2_latency"]:
            v = str(r[col]) if pd.notna(r[col]) else ""
            if v and not is_time(v):
                parts.append(v)
        for col in ["Q5j_other_reason", "Q10e_other_problem"]:
            v = str(r[col]) if pd.notna(r[col]) else ""
            if len(v.strip()) > 1:
                parts.append(v)
        return " / ".join(parts)

    text = txt.apply(joined, axis=1)
    out = pd.DataFrame({"Study_ID": txt["Study_ID"]})
    for name, pattern in TEXT_RULES.items():
        out[name] = text.str.contains(pattern, flags=re.I, regex=True).astype(int)
    return out


def main():
    raw = pd.read_excel(DATA_DIR / "TMD_Sleep_Cleaned_Dataset.xlsx", sheet_name="Data")
    txt = pd.read_csv(DATA_DIR / "psqi_freetext.csv", dtype=str)

    d = raw.merge(classify_schedule(txt), on="Study_ID", how="left")
    d = d[d["Repeat_patient"] == 0]
    d = d[pd.to_datetime(d["Study_date"]).dt.year <= 2023]
    d = d[d["Sleep_schedule"].notna()].copy()
    d = d.merge(text_flags(txt), on="Study_ID", how="left")

    d["Entry_quarter"] = pd.to_datetime(d["Study_date"]).dt.to_period("Q").astype(str)
    d["Non_regular"] = (d["Sleep_schedule"] != "Regular").astype(int)
    d["PSQI_core"] = d[["PSQI_C1", "PSQI_C2", "PSQI_C5", "PSQI_C6", "PSQI_C7"]].sum(axis=1, min_count=5)
    pct = np.where(d["VAS"] > 0, (d["VAS_final"] - d["VAS"]) / d["VAS"] * 100, np.nan)
    d["Responder30"] = pd.Series(pct <= -30, index=d.index).astype(float).where(~np.isnan(pct))
    d["Responder2pt"] = (d["VAS_change"] <= -2).astype(float).where(d["VAS_change"].notna())
    d["Tx_duration_mo"] = d["Tx_duration_m"].where(d["Tx_duration_m"] <= 60)

    columns = {
        "Study_ID": "Study_ID",
        "Study_date": "Visit_date",
        "Entry_quarter": "Entry_quarter",
        "Age": "Age",
        "Female": "Female",
        "BMI": "BMI",
        "Symptom_duration_m": "Symptom_duration_mo",
        "Chronic_ge6m": "Chronicity",
        "Bilateral_pain": "Bilateral_pain",
        "VAS": "VAS",
        "CMI_Expert": "CMI",
        "DI_Expert": "DI",
        "PI_Expert": "PI",
        "Locking_Expert": "Locking",
        "Noise_Expert": "Joint_noise",
        "MUO_mm": "MUO_mm",
        "HATMD": "HATMD",
        "Sleep_schedule": "Sleep_schedule",
        "Non_regular": "Non_regular",
        "Schedule_range_h": "Schedule_range_h",
        "Bedtime_rel_midnight": "Bedtime_h",
        "Wake_time": "Wake_time_h",
        "Sleep_duration_h": "Sleep_duration_h",
        "Sleep_latency_min": "Sleep_latency_min",
        **{f"PSQI_C{i}": f"PSQI_C{i}" for i in range(1, 8)},
        "PSQI_core": "PSQI_core",
        "PSQI_global": "PSQI_global",
        "Poor_sleeper": "Poor_sleeper",
        "STOP_score_0to4": "STOP_score",
        "CF_Bruxism": "Bruxism",
        "CF_Clenching": "Clenching",
        "CC_Stress": "Stress",
        "N_contrib_nonsleep": "Contributing_factors",
        "MH_any": "Systemic_disease",
        "Exclusion_flag": "Psych_or_pregnancy",
        "SCL_completed": "SCL_completed",
        "SCL_GSI_T": "GSI_T",
        "Sleep_bruxism_text": "Sleep_bruxism_text",
        "Pain_sleep_text": "Pain_sleep_text",
        "VAS_final": "VAS_post",
        "VAS_change": "dVAS",
        "Responder30": "Responder30",
        "Responder2pt": "Responder2pt",
        "Tx_finished": "Tx_completion",
        "Tx_duration_mo": "Tx_duration_mo",
        "Tx_Splint": "Splint",
        "Tx_Physical_therapy": "Physical_therapy",
        "Tx_Pharmacotherapy": "Pharmacotherapy",
    }
    out = d[list(columns)].rename(columns=columns).sort_values("Study_ID").reset_index(drop=True)
    for c in [
        "BMI",
        "Symptom_duration_mo",
        "CMI",
        "DI",
        "PI",
        "Schedule_range_h",
        "Bedtime_h",
        "Wake_time_h",
        "Sleep_duration_h",
        "Sleep_latency_min",
    ]:
        out[c] = pd.to_numeric(out[c]).round(3)
    out.to_excel(ANALYSIS_FILE, sheet_name="Data", index=False)
    print(f"{len(out)} patients; {out['Sleep_schedule'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
