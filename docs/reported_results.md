# Original reported results and reconstruction notes

The following figures are transcribed from the capstone presentation
`CP2 Presentation(2).pdf`, supported by the final report dated 3 August 2026.
They describe the submitted study. They are **not the output of the repository's
reconstructed implementation**.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9875 | 1.00 | 0.98 | 0.9899 | 1.00 |
| Decision Tree | 0.9875 | 1.00 | 0.98 | 0.9899 | 0.99 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| XGBoost | 0.9875 | 1.00 | 0.98 | 0.9899 | 0.9993 |
| LR + RF, soft voting | 0.9875 | 1.00 | 0.98 | 0.9899 | 1.00 |
| DT + XGB, soft voting | 0.9875 | 1.00 | 0.98 | 0.9899 | 1.00 |

The reported test set comprised 80 records: 50 CKD and 30 non-CKD. Random Forest
was the best-performing model in that reported evaluation.

## How the executable version was prepared

The original final corrected `.ipynb` or `.py` was unavailable. This repository
therefore reconstructs the workflow from the report, presentation, and supplied
CSV. It does not claim recovery of the exact final submitted source.

The report appendix specifies Logistic Regression with `max_iter=1000`, Decision
Tree with seed 42, Random Forest with 100 trees and seed 42, and XGBoost with
`eval_metric="logloss"` and seed 42. Both ensembles use equal-weight soft voting.
The stratified split uses 20% test data and seed 42.

The appendix shows imputation fitted before splitting. The repository instead
fits all learned preprocessing on training data only, following
[scikit-learn's guidance on data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
This addresses the observed preprocessing issue without asserting that the
unavailable final notebook still contained it.

Other implementation choices are explicit: the transformed feature order is the
14 numeric predictors followed by ten binary predictors; only the Logistic
Regression component is scaled inside LR + RF; and XGBoost uses the histogram
tree method, 100 estimators, and CPU execution with one worker. The original
package versions were not available, so the verified versions are pinned in
`requirements.txt` and recorded in `results/run_metadata.json`.

These differences can affect individual predictions and feature importance.
No model settings were tuned to force a match with the reported values.
Use [the current run's metrics](../results/metrics.csv) when describing the
executable repository, and label the table above as the original reported results.
