"""Regression checks for leakage prevention, target labels and soft voting."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ckd_pipeline import (
    CATEGORIES, NUMERIC_COLUMNS, clean_features, fit_models,
    load_dataset, split_dataset,
)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.DataFrame({
            column: [10, 20, np.nan, 40, 50, 60] for column in NUMERIC_COLUMNS
        })
        for column, categories in CATEGORIES.items():
            cls.raw[column] = [categories[1], categories[1], np.nan,
                               categories[1], categories[0], categories[1]]
        cls.X = clean_features(cls.raw)
        cls.y = pd.Series([0, 0, 1, 1, 0, 1])
        cls.models = fit_models(cls.X, cls.y)

    def test_dataset_and_disjoint_stratified_split(self):
        X, y = load_dataset()
        self.assertEqual(X.shape, (400, 24))
        self.assertNotIn("id", X.columns)
        self.assertNotIn("classification", X.columns)
        self.assertEqual(y.value_counts().to_dict(), {1: 250, 0: 150})
        train, test, train_y, test_y = split_dataset(X, y)
        self.assertEqual(len(train), 320)
        self.assertEqual(len(test), 80)
        self.assertFalse(set(train.index) & set(test.index))
        self.assertEqual(set(train.index) | set(test.index), set(X.index))
        self.assertEqual(test_y.value_counts().to_dict(), {1: 50, 0: 30})

    def test_imputation_and_scaling_use_training_records_only(self):
        heldout = self.raw.iloc[:2].copy()
        for column in NUMERIC_COLUMNS:
            heldout[column] = [np.nan, 1e9]
        for column, categories in CATEGORIES.items():
            heldout[column] = categories[0]
        heldout = clean_features(heldout)
        pipelines = list(self.models[key] for key in ["lr", "dt", "rf", "xgb"])
        for key in ["lr_rf", "dt_xgb"]:
            pipelines.extend(self.models[key].estimators_)
        for pipeline in pipelines:
            with self.subTest(estimator=type(pipeline.named_steps["model"]).__name__):
                preprocess = pipeline.named_steps["preprocess"]
                numeric = preprocess.named_transformers_["numeric"]
                categorical = preprocess.named_transformers_["categorical"].named_steps["imputer"]
                before_numeric = numeric.statistics_.copy()
                before_categorical = categorical.statistics_.copy()
                np.testing.assert_allclose(before_numeric, np.full(14, 40))
                self.assertEqual(before_categorical.tolist(), [v[1] for v in CATEGORIES.values()])
                self.assertEqual(preprocess.transform(heldout)[0, 0], 40)
                pipeline.predict_proba(heldout)
                np.testing.assert_array_equal(numeric.statistics_, before_numeric)
                np.testing.assert_array_equal(categorical.statistics_, before_categorical)
                if "scale" in pipeline.named_steps:
                    scaler = pipeline.named_steps["scale"]
                    self.assertEqual(scaler.n_samples_seen_, len(self.X))
                    self.assertAlmostEqual(scaler.mean_[0], np.mean([10, 20, 40, 40, 50, 60]))

    def test_soft_voting_averages_component_probabilities(self):
        for key in ["lr_rf", "dt_xgb"]:
            voting = self.models[key]
            expected = np.mean([p.predict_proba(self.X) for p in voting.estimators_], axis=0)
            np.testing.assert_allclose(voting.predict_proba(self.X), expected)
        self.assertIn("scale", self.models["lr_rf"].named_estimators_["lr"].named_steps)
        self.assertNotIn("scale", self.models["lr_rf"].named_estimators_["rf"].named_steps)

    def test_whitespace_and_missing_markers_are_cleaned(self):
        frame = self.raw.copy()
        frame.loc[0, "dm"] = "\t YES "
        frame["wc"] = frame["wc"].astype(object)
        frame.loc[0, "wc"] = "\t?"
        cleaned = clean_features(frame)
        self.assertEqual(cleaned.loc[0, "dm"], "yes")
        self.assertTrue(pd.isna(cleaned.loc[0, "wc"]))
        self.assertTrue(pd.isna(cleaned.loc[2, "age"]))

    def test_invalid_targets_and_categories_are_rejected(self):
        raw = self.raw.copy()
        raw["classification"] = ["ckd", "notckd", None, "ckd", "notckd", "ckd"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            raw.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Every target"):
                load_dataset(path)
        raw.loc[0, "dm"] = "maybe"
        with self.assertRaisesRegex(ValueError, "Unknown categories in dm"):
            clean_features(raw)

    def test_completely_missing_training_feature_is_rejected(self):
        X = self.X.copy()
        X["age"] = np.nan
        with self.assertRaisesRegex(ValueError, "no observed values"):
            fit_models(X, self.y)


if __name__ == "__main__":
    unittest.main()
