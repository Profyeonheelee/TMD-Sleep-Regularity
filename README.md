# Irregular sleep, jaw dysfunction and pain in temporomandibular disorders

Analysis code for:

> Irregular Sleep Is Associated with Jaw Dysfunction but Not Pain in Temporomandibular Disorders.

Statistical analyses and tables were run in Python; figures were drawn in R from the
figure data written by the Python scripts.

## Data

The patient-level data are not included because of patient confidentiality (IRB No. KH-DT25033,
Kyung Hee University Dental Hospital). De-identified data are available from the corresponding
author on reasonable request. Place the files in `data/`:

| File | Content |
|---|---|
| `TMD_Sleep_Cleaned_Dataset.xlsx` | De-identified first-visit records (sheet `Data`) |
| `psqi_freetext.csv` | Verbatim PSQI answers: `Study_ID`, `Q1_bedtime`, `Q2_latency`, `Q3_waketime`, `Q4_sleep_hours`, `Q5j_other_reason`, `Q10e_other_problem` |

`prepare_data.py` creates `data/analysis_dataset.xlsx`, which all other scripts use.

## Structure

```
python/
  common.py              paths, derived variables, Model 1-4 covariate sets, helpers
  gated_transformer.py   gated attention feature-tokenizer Transformer (scikit-learn interface)
  prepare_data.py        cohort selection, sleep-schedule classification, free-text flags
  Table1.py              Table 1 and Supplementary Table S1
  Table2.py              Table 2 and Supplementary Table S2
  TableS3.py             Supplementary Table S3 (cross-validated performance)
  Figure1_data.py ... FigureS5_data.py   data for each figure -> results/Figure_data.xlsx
R/
  Figure1.R ... FigureS5.R               figures -> figures/
```

## Running

Run from the repository root.

```bash
pip install -r requirements.txt
cd python
python prepare_data.py
python Table1.py
python Table2.py
python TableS3.py          # needed before FigureS2_data.py and FigureS3_data.py
for f in Figure1 Figure2 Figure3 Figure4 Figure5 FigureS1 FigureS2 FigureS3 FigureS4 FigureS5; do
  python ${f}_data.py
done
cd ..
for f in R/Figure*.R; do Rscript "$f"; done
```

R packages: `readxl`, `dplyr`, `tidyr`, `ggplot2`, `patchwork`, `scales`.

Bootstrap, permutation, cross-fitting and cross-validation use fixed random seeds. The Transformer
models (Figure 4, Figure 5, Table S3) and the causal forest (Figure S4) are computationally
intensive; results of neural-network models may differ slightly across hardware and library versions.

## Software

Python 3.11 (numpy 2.4, pandas 3.0, scipy 1.17, statsmodels 0.15, scikit-learn 1.8, scikit-posthocs,
torch 2.14, shap 0.51, econml 0.17) and R 4.3 (ggplot2, patchwork).
