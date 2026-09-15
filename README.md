# Predicting Chronic Kidney Disease Risk Using Machine Learning Techniques

**Author:** [Tan Yann Bin](https://github.com/yannbintan)  
**Institution:** Sunway University  
**Project:** Capstone Project 2, Data Analytics

An academic machine learning study comparing four individual classifiers and two soft-voting ensembles for Chronic Kidney Disease (CKD) classification. The project combines exploratory data analysis, model evaluation, and interpretability using structured clinical records.

> **Repository preparation status:** This README is prepared from the submitted capstone report and presentation. The final corrected notebook or Python script is still required before executable code, dependency versions, and verified run instructions can be added.

## Project objectives

- Identify clinical variables that contribute to the models' CKD predictions.
- Develop and compare individual classifiers and soft-voting ensembles.
- Assess classification performance alongside interpretability.

## Dataset

The supplied `kidney_disease.csv` contains **400 records** and **26 columns**:

- One identifier column, `id`, excluded from modelling.
- **24 predictor columns** containing numeric and categorical measurements.
- One target column, `classification`.

After removing `id`, 25 columns remain, including the target. The target is also excluded from the predictor matrix. After trimming label whitespace, the supplied data contains **250 CKD** and **150 non-CKD** records. The project uses **CKD = 1** and **non-CKD = 0**.

Predictors include age, blood pressure, specific gravity, albumin, sugar, blood glucose, blood urea, serum creatinine, sodium, potassium, hemoglobin, blood cell measurements, and categorical clinical indicators. The dataset includes missing values and inconsistent text formatting.

The study identifies the dataset as the public CKD dataset obtained through Kaggle. Its exact source URL and redistribution information should accompany the final code package.

## Methodology

The project workflow covers:

1. Inspecting the dataset, removing the identifier, cleaning text labels, and converting numeric fields.
2. Exploring numeric distributions, categorical frequencies, class balance, and correlations.
3. Preparing training and test sets for classification.
4. Handling missing values, encoding categorical variables, and applying scaling where required.
5. Training four individual models and two soft-voting ensembles.
6. Evaluating predictions and examining model explanations.

The corrected implementation must split the data **before fitting imputers or scalers**. Imputation and scaling parameters must be learned from the training set and then applied to the test set. Scaling is required for the Logistic Regression component; the standalone tree models do not require it. The final source file is needed to verify these steps and the exact ensemble preprocessing.

The report's code appendix shows an earlier preprocessing order, with imputation fitted before the train/test split. Those screenshots are not being treated as the final corrected executable implementation.

## Models and reported results

The values below are the **reported capstone results**, reproduced from the submitted presentation. They have not been regenerated while preparing this repository.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9875 | 1.00 | 0.98 | 0.9899 | 1.00 |
| Decision Tree | 0.9875 | 1.00 | 0.98 | 0.9899 | 0.99 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| XGBoost | 0.9875 | 1.00 | 0.98 | 0.9899 | 0.9993 |
| Logistic Regression + Random Forest, soft voting | 0.9875 | 1.00 | 0.98 | 0.9899 | 1.00 |
| Decision Tree + XGBoost, soft voting | 0.9875 | 1.00 | 0.98 | 0.9899 | 1.00 |

The reported test set contains **80 records: 50 CKD and 30 non-CKD**. Random Forest correctly classified all 80 records in that reported evaluation. The other five models each correctly classified 79 records.

The two hybrid models are **soft-voting ensembles**, combining predicted probabilities from their constituent classifiers. They should not be described as sequential feature-selection pipelines.

## Interpretability

The study reports Logistic Regression coefficients, tree-based feature importance, and SHAP analysis. Frequently discussed predictors include serum creatinine, hemoglobin, packed cell volume, specific gravity, red blood cell count, albumin, hypertension, and diabetes mellitus.

These explanations describe model behaviour and associations within the dataset. They do not establish causal effects.

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn, XGBoost, and SHAP.

Exact package versions and installation commands will be recorded after inspecting the final executable source.

## My contribution

I developed this capstone project from problem definition and dataset preparation through model comparison, interpretation, and reporting. My work included exploratory analysis, handling missing and inconsistent values, developing the six model configurations, comparing evaluation metrics, and explaining model predictions using coefficients, feature importance, and SHAP.

## Reproduction and validation status

The dataset's dimensions and class counts have been checked. The final corrected `.ipynb` or `.py` file is still needed to complete the repository and verify the preprocessing, dependencies, model settings, and reported results. No model training or end-to-end execution has been performed for this repository preparation.

The reported results come from a small dataset and a single 80-record test set. Further validation on independent and more diverse data is needed. This is an academic classification study and has not been validated for clinical use.

## Documentation basis

- Capstone Project 2 final report, dated 3 August 2026.
- Final presentation supplied as `CP2 Presentation(2).pdf`.
- The supplied `kidney_disease.csv` dataset.

The report and presentation document the completed research. The final corrected source code remains the required basis for a reproducible implementation.
