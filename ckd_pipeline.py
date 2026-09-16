"""Reconstructed CKD capstone workflow; run with `python ckd_pipeline.py`.

This is a repository implementation based on the capstone report, not the
unavailable original final notebook. Reported metrics are never used in training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "kidney_disease.csv"
NUMERIC_COLUMNS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc",
]
# Fixed, documented categories: the first value maps to 0 and the second to 1.
# Imputation still happens on the original labels, using training-set modes.
CATEGORIES = {
    "rbc": ["abnormal", "normal"],
    "pc": ["abnormal", "normal"],
    "pcc": ["notpresent", "present"],
    "ba": ["notpresent", "present"],
    "htn": ["no", "yes"],
    "dm": ["no", "yes"],
    "cad": ["no", "yes"],
    "appet": ["poor", "good"],
    "pe": ["no", "yes"],
    "ane": ["no", "yes"],
}
FEATURE_COLUMNS = NUMERIC_COLUMNS + list(CATEGORIES)
FEATURE_LABELS = dict(zip(FEATURE_COLUMNS, [
    "Age", "Blood pressure", "Specific gravity", "Albumin", "Sugar",
    "Blood glucose", "Blood urea", "Serum creatinine", "Sodium", "Potassium",
    "Hemoglobin", "Packed cell volume", "White blood cell count",
    "Red blood cell count", "Red blood cells", "Pus cell", "Pus cell clumps",
    "Bacteria", "Hypertension", "Diabetes mellitus", "Coronary artery disease",
    "Appetite", "Pedal edema", "Anemia",
]))
MODEL_NAMES = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lr_rf": "LR + RF (soft voting)",
    "dt_xgb": "DT + XGB (soft voting)",
}


def normalize_text(values: pd.Series) -> pd.Series:
    """Clean fixed formatting without learning anything from other records."""
    return (
        values.astype("string").str.lower()
        .str.replace(r"\s+", "", regex=True)
        .replace(["", "?", "nan", "none", "null"], pd.NA)
    )


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the schema and clean values; do not impute or scale here."""
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing predictor columns: {', '.join(missing)}")
    cleaned = pd.DataFrame(index=frame.index)
    for column in NUMERIC_COLUMNS:
        text = normalize_text(frame[column])
        values = pd.to_numeric(text, errors="coerce")
        invalid = text.notna() & values.isna()
        if invalid.any():
            raise ValueError(f"Invalid numeric values in {column}: {text[invalid].unique().tolist()}")
        cleaned[column] = values.astype(float)
        if np.isinf(cleaned[column]).any():
            raise ValueError(f"Infinite numeric value in {column}")
    for column, categories in CATEGORIES.items():
        values = normalize_text(frame[column])
        invalid = values.notna() & ~values.isin(categories)
        if invalid.any():
            raise ValueError(f"Unknown categories in {column}: {values[invalid].unique().tolist()}")
        cleaned[column] = values.to_numpy(dtype=object, na_value=np.nan)
    return cleaned[FEATURE_COLUMNS]


def load_dataset(path: str | Path = DEFAULT_DATA) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, dtype=str)
    if "classification" not in frame:
        raise ValueError("The CSV must contain a classification target column.")
    labels = normalize_text(frame["classification"])
    if labels.isna().any() or not labels.isin(["ckd", "notckd"]).all():
        raise ValueError("Every target must be ckd or notckd; missing targets are not imputed.")
    target = labels.map({"notckd": 0, "ckd": 1}).astype(int)
    if target.nunique() != 2:
        raise ValueError("Both CKD and non-CKD records are required for evaluation.")
    # Explicit feature selection excludes both id and classification.
    return clean_features(frame), target


def split_dataset(X, y, *, seed: int = 42, test_size: float = 0.2):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), NUMERIC_COLUMNS),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OrdinalEncoder(categories=list(CATEGORIES.values()))),
        ]), list(CATEGORIES)),
    ], verbose_feature_names_out=False)


def make_model(estimator, *, scale: bool = False) -> Pipeline:
    steps = [("preprocess", make_preprocessor())]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps + [("model", estimator)])


def build_models(seed: int = 42) -> dict:
    """Report settings, with a separate preprocessing pipeline for every learner."""
    models = {
        "lr": make_model(LogisticRegression(max_iter=1000, random_state=seed), scale=True),
        "dt": make_model(DecisionTreeClassifier(random_state=seed)),
        "rf": make_model(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)),
        "xgb": make_model(XGBClassifier(
            n_estimators=100, eval_metric="logloss", random_state=seed,
            tree_method="hist", n_jobs=1,
        )),
    }
    # VotingClassifier clones and fits these entire pipelines on X_train only.
    # The forest in LR + RF receives unscaled data; only LR has a scaler.
    models["lr_rf"] = VotingClassifier(
        estimators=[("lr", clone(models["lr"])), ("rf", clone(models["rf"]))],
        voting="soft", n_jobs=1,
    )
    models["dt_xgb"] = VotingClassifier(
        estimators=[("dt", clone(models["dt"])), ("xgb", clone(models["xgb"]))],
        voting="soft", n_jobs=1,
    )
    return models


def fit_models(X_train, y_train, *, seed: int = 42) -> dict:
    empty = X_train.columns[X_train.isna().all()].tolist()
    if empty:
        raise ValueError(f"Training columns have no observed values: {empty}")
    models = build_models(seed)
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models, X_test, y_test):
    rows, predictions, reports = [], [], {}
    for key, model in models.items():
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_test)[:, positive_index]
        predicted = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
        rows.append({
            "model": MODEL_NAMES[key],
            "accuracy": accuracy_score(y_test, predicted),
            "precision": precision_score(y_test, predicted, zero_division=0),
            "recall": recall_score(y_test, predicted, zero_division=0),
            "f1": f1_score(y_test, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
        predictions.append(pd.DataFrame({
            "model": MODEL_NAMES[key], "row_index": X_test.index,
            "actual": y_test.to_numpy(), "predicted": predicted,
            "ckd_probability": probabilities,
        }))
        reports[key] = classification_report(
            y_test, predicted, labels=[0, 1], target_names=["non-CKD", "CKD"],
            output_dict=True, zero_division=0,
        )
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True), reports


def save_figure(fig, directory: Path, name: str):
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def save_eda(X_train, y_train, y_test, directory: Path):
    """Feature EDA uses training records; class counts also show the test split."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = pd.DataFrame({
        "Training": y_train.value_counts().reindex([0, 1]),
        "Test": y_test.value_counts().reindex([0, 1]),
    }, index=[0, 1])
    counts.index = ["Non-CKD (0)", "CKD (1)"]
    counts.plot.bar(stacked=True, ax=ax, color=["#187b80", "#e8a34d"], rot=0)
    ax.set(title="Class balance by split", ylabel="Records", xlabel="")
    save_figure(fig, directory, "class_balance")

    fig, ax = plt.subplots(figsize=(8, 6))
    missing = X_train.isna().sum().sort_values()
    ax.barh([FEATURE_LABELS[c] for c in missing.index], missing, color="#187b80")
    ax.set(title="Missing values in training records", xlabel="Missing records")
    save_figure(fig, directory, "missing_values")

    fig, axes = plt.subplots(4, 4, figsize=(14, 11), layout="constrained")
    for column, ax in zip(NUMERIC_COLUMNS, axes.flat):
        ax.hist(X_train[column].dropna(), bins=15, color="#187b80", edgecolor="white")
        ax.set(title=FEATURE_LABELS[column], ylabel="Records")
    for ax in list(axes.flat)[len(NUMERIC_COLUMNS):]:
        ax.set_visible(False)
    fig.suptitle("Numeric feature distributions (training data, before imputation)")
    save_figure(fig, directory, "numeric_distributions")

    fig, axes = plt.subplots(2, 5, figsize=(15, 7), layout="constrained")
    for column, ax in zip(CATEGORIES, axes.flat):
        counts = X_train[column].fillna("missing").value_counts()
        ax.bar(counts.index, counts, color="#187b80")
        ax.set(title=FEATURE_LABELS[column], ylabel="Records")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Categorical feature distributions (training data)")
    save_figure(fig, directory, "categorical_distributions")

    corr = X_train[NUMERIC_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    heat = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [FEATURE_LABELS[c] for c in NUMERIC_COLUMNS]
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Numeric correlations (training data, pairwise complete values)")
    fig.colorbar(heat, ax=ax, label="Pearson correlation", shrink=0.8)
    save_figure(fig, directory, "numeric_correlations")


def save_evaluation_plots(metrics, predictions, directory: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(metrics))
    ax.barh(positions - 0.18, metrics["accuracy"], 0.36, label="Accuracy", color="#187b80")
    ax.barh(positions + 0.18, metrics["roc_auc"], 0.36, label="ROC-AUC", color="#e8a34d")
    ax.set_yticks(positions, metrics["model"])
    ax.set(xlim=(0, 1.06), xlabel="Score", title="Reconstructed implementation: test performance")
    ax.legend(loc="lower left")
    save_figure(fig, directory, "model_comparison")

    fig, ax = plt.subplots(figsize=(8, 6))
    for model, values in predictions.groupby("model", sort=False):
        fpr, tpr, _ = roc_curve(values["actual"], values["ckd_probability"])
        auc = roc_auc_score(values["actual"], values["ckd_probability"])
        ax.plot(fpr, tpr, label=f"{model} (AUC={auc:.4f})", linewidth=1.8)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="False positive rate",
           ylabel="True positive rate", title="Test ROC curves (positive class: CKD)")
    ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, directory, "roc_curves")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for (model, values), ax in zip(predictions.groupby("model", sort=False), axes.flat):
        ConfusionMatrixDisplay.from_predictions(
            values["actual"], values["predicted"], labels=[0, 1],
            display_labels=["Non-CKD", "CKD"], cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(model, fontsize=10)
    save_figure(fig, directory, "confusion_matrices")


def save_feature_importance(models, directory: Path):
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    for key, ax in zip(["lr", "dt", "rf", "xgb"], axes.flat):
        pipeline = models[key]
        features = pipeline.named_steps["preprocess"].get_feature_names_out()
        estimator = pipeline.named_steps["model"]
        values = estimator.coef_[0] if key == "lr" else estimator.feature_importances_
        measure = "standardized coefficient" if key == "lr" else "feature importance"
        for feature, value in zip(features, values):
            rows.append({"model": MODEL_NAMES[key], "feature": feature,
                         "measure": measure, "value": float(value)})
        indices = np.argsort(np.abs(values))[-10:]
        ax.barh([FEATURE_LABELS[features[i]] for i in indices], values[indices], color="#187b80")
        ax.set(title=MODEL_NAMES[key], xlabel=measure.capitalize())
        ax.axvline(0, color="grey", linewidth=0.7)
    save_figure(fig, directory, "feature_importance")
    pd.DataFrame(rows).to_csv(directory.parent / "feature_importance.csv", index=False)


def save_shap(models, X_test, directory: Path):
    import shap

    pipeline = models["rf"]
    preprocessor = pipeline.named_steps["preprocess"]
    features = preprocessor.get_feature_names_out()
    test = pd.DataFrame(preprocessor.transform(X_test), columns=features, index=X_test.index)
    # Use training path counts stored in the forest as the background distribution.
    # For sklearn RandomForestClassifier, raw class outputs are probabilities.
    explainer = shap.TreeExplainer(
        pipeline.named_steps["model"],
        feature_perturbation="tree_path_dependent", model_output="raw",
    )
    explanation = explainer(test, check_additivity=True)
    # RandomForestClassifier produces one explanation per class.
    positive = explanation[:, :, list(pipeline.classes_).index(1)]
    expected = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
    np.testing.assert_allclose(
        positive.base_values + positive.values.sum(axis=1), expected, atol=1e-6,
    )
    summary = pd.DataFrame({
        "feature": features, "mean_absolute_shap": np.abs(positive.values).mean(axis=0),
    }).sort_values("mean_absolute_shap", ascending=False)
    summary.to_csv(directory.parent / "shap_importance.csv", index=False)
    values = pd.DataFrame(positive.values, index=X_test.index, columns=features)
    values.insert(0, "base_probability", positive.base_values)
    values.rename_axis("row_index").to_csv(directory.parent / "shap_values_ckd.csv")
    best = summary.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([FEATURE_LABELS[f] for f in best["feature"]], best["mean_absolute_shap"], color="#187b80")
    ax.set(title="Random Forest: CKD probability explanations on test records",
           xlabel="Mean absolute SHAP value (probability units)")
    save_figure(fig, directory, "shap_importance")
    return {"model": "Random Forest", "positive_class": 1,
            "background": "training path counts stored in the fitted forest",
            "feature_perturbation": "tree_path_dependent",
            "model_output": "raw class probabilities (sklearn RandomForestClassifier)",
            "additivity_check": "passed", "test_records": len(X_test),
            "max_probability_reconstruction_error": float(np.abs(positive.base_values + positive.values.sum(axis=1) - expected).max())}


def package_versions():
    versions = {}
    for name in ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "xgboost", "xgboost-cpu", "shap", "numba", "llvmlite"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def json_safe(value):
    """Keep model settings valid JSON, including XGBoost's NaN missing marker."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def run(data_path=DEFAULT_DATA, output_dir="outputs", *, seed=42, test_size=0.2, with_shap=True):
    data_path, output_dir = Path(data_path), Path(output_dir)
    # Use a fresh directory so a new or failed run cannot masquerade as old results.
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output path is not an empty directory: {output_dir}. Choose a new --output directory.")
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_dataset(X, y, seed=seed, test_size=test_size)
    models = fit_models(X_train, y_train, seed=seed)
    metrics, predictions, reports = evaluate_models(models, X_test, y_test)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "classification_reports.json").write_text(json.dumps(reports, indent=2) + "\n")
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "ckd-capstone", "font.size": 9})
    save_eda(X_train, y_train, y_test, figures)
    save_evaluation_plots(metrics, predictions, figures)
    save_feature_importance(models, figures)
    shap_status = save_shap(models, X_test, figures) if with_shap else {"status": "skipped by request"}
    metadata = {
        "status": "completed", "implementation": "reconstructed from capstone documentation",
        "dataset_file": data_path.name,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": len(X), "predictors": len(X.columns), "target": {"non-CKD": 0, "CKD": 1},
        "seed": seed, "test_size": test_size, "train_records": len(X_train), "test_records": len(X_test),
        "train_class_counts": y_train.value_counts().sort_index().to_dict(),
        "test_class_counts": y_test.value_counts().sort_index().to_dict(),
        "train_row_indices": X_train.index.tolist(), "test_row_indices": X_test.index.tolist(),
        "preprocessing": "fit only on training data, separately within every model pipeline",
        "model_parameters": {key: models[key].named_steps["model"].get_params() for key in ["lr", "dt", "rf", "xgb"]},
        "ensembles": {"lr_rf": ["lr", "rf"], "dt_xgb": ["dt", "xgb"], "voting": "soft", "weights": "equal"},
        "shap": shap_status, "python": platform.python_version(), "platform": platform.system(),
        "packages": package_versions(),
    }
    # Written last: an interrupted or failed run has no completed-run marker.
    (output_dir / "run_metadata.json").write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CKD CSV with the documented schema")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="New or empty output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--skip-shap", action="store_true", help="Skip the optional Random Forest SHAP analysis")
    args = parser.parse_args()
    try:
        metrics = run(args.data, args.output, seed=args.seed, test_size=args.test_size, with_shap=not args.skip_shap)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nCompleted. Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
