"""Figure 1: study flow, population actogram and timing tests."""

import re

import numpy as np
import pandas as pd

from common import DATA_DIR, RESULTS, load_data, write_sheets
from prepare_data import classify_schedule


def endpoints(text):
    """First and last value of a reported time or range ('23~1' -> 23, 1); NaN if not numeric."""
    if not isinstance(text, str):
        return np.nan, np.nan
    u = text.replace(" ", "")
    hm = re.findall(r"(\d{1,2}):(\d{2})", u)
    vals = [int(a) + int(b) / 60 for a, b in hm] if hm else [float(x) for x in re.findall(r"\d+(?:\.\d+)?", u)]
    if not vals:
        return np.nan, np.nan
    if len(vals) >= 2 and re.search(r"[~\-,/]", u):
        return vals[0], vals[1]
    return vals[0], vals[0]


def bed_clock(h):
    """Bedtime on a continuous clock (24 = midnight, 25 = 01:00); afternoon answers are not interpretable."""
    if 18 <= h <= 30:
        return h
    if 0 <= h <= 7:
        return 24 + h
    if 7 < h <= 12:
        return 12 + h
    return np.nan


def bed_range(text):
    a, b = (bed_clock(v) if not np.isnan(v) else np.nan for v in endpoints(text))
    return a, (a if b < a else b)


def wake_range(text):
    a, b = (v + 24 for v in endpoints(text))
    return a, (b + 12 if b < a else b)


def flow(d):
    raw = pd.read_excel(DATA_DIR / "TMD_Sleep_Cleaned_Dataset.xlsx", sheet_name="Data")
    txt = pd.read_csv(DATA_DIR / "psqi_freetext.csv", dtype=str)
    raw = raw.merge(classify_schedule(txt), on="Study_ID", how="left")
    repeat = raw[raw.Repeat_patient == 1]
    unique = raw[raw.Repeat_patient == 0]
    early = unique[pd.to_datetime(unique.Study_date).dt.year <= 2023]
    n = d.Sleep_schedule.value_counts()
    return pd.DataFrame(
        [
            ["First-visit records", "", len(raw)],
            [
                "Excluded: repeat visits",
                f"{repeat.Patient_key.nunique()} patients ({len(repeat)} records)",
                len(repeat),
            ],
            ["Unique patients", "", len(unique)],
            ["Excluded: 2024 entries", "", len(unique) - len(early)],
            ["Patients 2021-2023", "", len(early)],
            ["Excluded: schedule not determinable", "", int(early.Sleep_schedule.isna().sum())],
            ["Analytic cohort", "", len(d)],
            ["Regular sleepers", "", int(n["Regular"])],
            ["Irregular sleepers", "", int(d.Non_regular.sum())],
            ["Variable timing", "", int(n["Variable"])],
            ["No fixed schedule", "", int(n["Irregular"])],
            ["Follow-up VAS available", "", int(d.dVAS.notna().sum())],
        ],
        columns=["Box", "Note", "n"],
    )


def actogram(d):
    txt = pd.read_csv(DATA_DIR / "psqi_freetext.csv", dtype=str)
    a = d[
        [
            "Study_ID",
            "Group",
            "Subgroup",
            "Sleep_schedule",
            "Bedtime_h",
            "Wake_time_h",
            "Midsleep_h",
            "Sleep_duration_h",
        ]
    ].merge(txt, on="Study_ID", how="left")
    bed = pd.DataFrame(a.Q1_bedtime.apply(bed_range).tolist(), columns=["lo", "hi"])
    wake = pd.DataFrame(a.Q3_waketime.apply(wake_range).tolist(), columns=["lo", "hi"])
    out = pd.DataFrame(
        {
            "Study_ID": a.Study_ID,
            "Group": np.where(a.Group == "Regular", "Regular sleepers", "Irregular sleepers"),
            "Subgroup": a.Subgroup.replace({"Regular": "Regular sleepers"}),
            "bed_lo": bed.lo,
            "bed_hi": bed.hi,
            "wake_lo": wake.lo,
            "wake_hi": wake.hi,
            "Midsleep_h": a.Midsleep_h,
            "Sleep_duration_h": a.Sleep_duration_h,
        }
    )
    return out


def timing_tests():
    t1 = pd.read_excel(RESULTS / "Table1.xlsx", sheet_name="Table1")
    rows = {
        "Bedtime": "Bedtime, clock time",
        "Wake time": "Wake time, clock time",
        "Mid-sleep": "Mid-sleep, clock time",
    }
    out = []
    for marker, label in rows.items():
        r = t1[t1.Characteristic == label].iloc[0]
        out.append([marker, r.p, "Welch t test", r.q])
    return pd.DataFrame(out, columns=["Marker", "p", "Test", "q_FDR"])


def main():
    d = load_data()
    write_sheets({"Fig1a_flow": flow(d), "Fig1b_actogram": actogram(d), "Fig1d_tests": timing_tests()})


if __name__ == "__main__":
    main()
