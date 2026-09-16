# Dataset attribution and preparation

`kidney_disease.csv` is the CSV supplied for this capstone and is included unchanged.
It contains 400 rows and 26 columns: an identifier, 24 predictors, and the target
`classification`. Labels become CKD = 1 and non-CKD = 0 in the code.

## Original source

Rubini, L., Soundarapandian, P., & Eswaran, P. (2015).
*Chronic Kidney Disease* [Dataset]. UCI Machine Learning Repository.
[DOI: 10.24432/C5G020](https://doi.org/10.24432/C5G020).

The [UCI dataset page](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
lists the dataset under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
Credit for the dataset belongs to its original creators. No endorsement is implied.

The capstone report describes obtaining the data from Kaggle but does not preserve
the exact download URL. To verify provenance, the supplied file was compared with
[UCI's CSV](https://archive.ics.uci.edu/static/public/336/data.csv): all 9,600
predictor cells and all 400 targets match after normalizing whitespace, missing
markers and numeric formatting. UCI's `wbcc`, `rbcc`, and `class` correspond to
`wc`, `rc`, and `classification`; the supplied CSV also includes `id`.

## File integrity

- Size: 48,551 bytes.
- SHA-256: `835c57ec100db2d3751165fae615fec425caf444cca23bab6a853d4c08bf5556`.
- Class counts after whitespace cleanup: 250 CKD, 150 non-CKD.
- The raw file keeps its original missing values and formatting.

## Modelling choices

`id` and `classification` are excluded from predictors. Fourteen columns are
represented numerically, including specific gravity (`sg`), albumin (`al`), and
sugar (`su`), following the capstone implementation. UCI describes those three as
nominal/ordinal measurements; their numeric treatment is a modelling choice.
The remaining ten predictors use the fixed binary mappings documented in
`CATEGORIES` in `ckd_pipeline.py`.

Numeric missing values use training medians; categorical missing values use
training modes. No imputation or scaling is applied to the CSV on disk.
