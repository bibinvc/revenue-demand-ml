"""Data, leakage, metric and artifact invariants for the synthetic experiment."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
from sqlalchemy import create_engine, text
import torch

from experiment import (Config, Preprocessor, SCHEMA, causal_design, classification_metrics,
                        deterministic, drift_metrics, fit_drift, make_data, paired_rmse_interval,
                        partialling_out, positive_weight, regression_metrics, ridge_fit,
                        ridge_predict, run, temporal_split, train_mlp)


class DataAndMethodTests(unittest.TestCase):
    def setUp(self):
        deterministic(18)
        self.config = Config(days=40, regression_steps=5, classification_steps=5, bootstrap_samples=20)
        self.data = make_data(self.config)
        self.splits = temporal_split(self.data)

    def test_generator_is_deterministic(self):
        pd.testing.assert_frame_equal(self.data, make_data(self.config))

    def test_splits_are_temporally_ordered_and_disjoint(self):
        train, val, test = (self.splits[k] for k in ("train", "validation", "test"))
        self.assertLess(train.day.max(), val.day.min())
        self.assertLess(val.day.max(), test.day.min())
        sets = [set(p.row_id) for p in self.splits.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(len(p) for p in self.splits.values()), len(self.data))

    def test_shifted_source_ids_are_separate(self):
        shifted = make_data(self.config, shifted=True)
        self.assertFalse(set(shifted.row_id) & set(self.data.row_id))
        self.assertTrue((shifted.source == "shifted").all())

    def test_split_rejects_too_few_days(self):
        with self.assertRaises(ValueError):
            temporal_split(self.data.loc[self.data.day < 10])

    def test_schema_rejects_duplicate_identifiers(self):
        broken = self.data.copy()
        broken.loc[1, "row_id"] = broken.loc[0, "row_id"]
        with self.assertRaises(pa.errors.SchemaError):
            SCHEMA.validate(broken)

    def test_schema_rejects_nonpositive_price(self):
        broken = self.data.copy()
        broken.loc[0, "price"] = 0.0
        with self.assertRaises(pa.errors.SchemaError):
            SCHEMA.validate(broken)

    def test_schema_rejects_infinite_values(self):
        broken = self.data.copy()
        broken.loc[0, "demand"] = np.inf
        with self.assertRaises(pa.errors.SchemaError):
            SCHEMA.validate(broken)

    def test_schema_rejects_unknown_extra_fields(self):
        with self.assertRaises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            SCHEMA.validate(self.data.assign(secret_target=1))

    def test_preprocessing_learns_only_training_rows(self):
        train, test = self.splits["train"], self.splits["test"]
        prep = Preprocessor().fit(train)
        before_mean, before_median = prep.mean.copy(), prep.market_median
        poisoned = test.copy()
        poisoned["market"], poisoned["demand"] = 100000.0, 999999.0
        prep.transform(poisoned)
        pd.testing.assert_series_equal(before_mean, prep.mean)
        self.assertEqual(before_median, prep.market_median)
        self.assertEqual(set(prep.fit_ids), set(train.row_id))

    def test_targets_and_identifiers_are_not_features(self):
        prep = Preprocessor().fit(self.splits["train"])
        self.assertFalse({"demand", "row_id", "source", "high_demand_label"} & set(prep.columns))

    def test_missing_covariates_are_imputed_finitely(self):
        prep = Preprocessor().fit(self.splits["train"])
        x = self.splits["test"].copy()
        x["market"] = np.nan
        transformed = prep.transform(x)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.dtype, np.float32)

    def test_all_missing_training_covariate_is_rejected(self):
        train = self.splits["train"].copy()
        train["market"] = np.nan
        with self.assertRaises(ValueError):
            Preprocessor().fit(train)

    def test_feature_engineering_is_mathematically_explicit(self):
        prep = Preprocessor().fit(self.splits["train"])
        features = prep.features(self.data)
        np.testing.assert_allclose(features.log_price, np.log(self.data.price))
        np.testing.assert_allclose(features.weekly_sin, np.sin(2 * np.pi * self.data.day / 7))
        self.assertTrue(any(c.startswith("route_") for c in features))

    def test_regression_metrics_match_hand_calculation(self):
        values = regression_metrics(np.array([1., 3.]), np.array([2., 1.]))
        self.assertAlmostEqual(values["rmse"], math.sqrt(2.5))
        self.assertEqual(values["mae"], 1.5)

    def test_classification_confusion_counts_and_metrics(self):
        result = classification_metrics(np.array([1, 1, 0, 0]), np.array([.8, .2, .6, .1]))
        self.assertEqual([result[k] for k in ("tp", "tn", "fp", "fn")], [1, 1, 1, 1])
        for key in ("accuracy", "precision", "recall", "f1"):
            self.assertEqual(result[key], 0.5)

    def test_classification_zero_division_is_defined(self):
        result = classification_metrics(np.array([0, 0]), np.array([.1, .2]))
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["recall"], 0)

    def test_positive_weight_is_training_class_ratio(self):
        self.assertEqual(positive_weight(np.array([0, 0, 0, 1])), 3)
        for invalid in (np.array([0, 0]), np.array([0, 0.4, 1])):
            with self.assertRaises(ValueError):
                positive_weight(invalid)

    def test_paired_bootstrap_is_seeded_and_identity_is_zero(self):
        y = np.arange(8, dtype=float)
        first = paired_rmse_interval(y, y + 1, y + 1, 18, 100)
        self.assertEqual(first["difference"], 0)
        self.assertEqual(first["ci95"], [0, 0])
        self.assertEqual(first, paired_rmse_interval(y, y + 1, y + 1, 18, 100))

    def test_ridge_baseline_reproduces_linear_fixture(self):
        x = np.array([[0.], [1.], [2.], [3.]])
        y = 2 + 3 * x[:, 0]
        coefficients = ridge_fit(x, y, penalty=0)
        np.testing.assert_allclose(ridge_predict(coefficients, x), y, atol=1e-9)

    def test_drift_identity_and_changed_distribution(self):
        values = np.linspace(-2, 2, 100)
        reference = fit_drift(values)
        before_edges = reference["edges"].copy()
        self.assertAlmostEqual(drift_metrics(reference, values)["psi"], 0)
        changed = drift_metrics(reference, values + 3)
        self.assertGreater(changed["psi"], .2)
        self.assertGreater(changed["ks_statistic"], .5)
        self.assertFalse(changed["threshold_is_statistical_test"])
        np.testing.assert_array_equal(before_edges, reference["edges"])

    def test_constant_reference_and_invalid_drift_inputs(self):
        reference = fit_drift(np.ones(10))
        self.assertEqual(drift_metrics(reference, np.ones(10))["ks_statistic"], 0)
        with self.assertRaises(ValueError):
            fit_drift(np.array([np.nan]))
        with self.assertRaises(ValueError):
            drift_metrics(reference, np.array([]))

    def test_causal_features_exclude_treatment_and_outcome(self):
        complete = self.data.dropna(subset=["market"])
        before = causal_design(complete)
        after = causal_design(complete.assign(price=1.0, demand=9000.0))
        np.testing.assert_array_equal(before, after)

    def test_causal_folds_are_disjoint_and_cover_complete_cases(self):
        result = partialling_out(self.splits["train"], 18)
        ids = []
        for fold in result["fold_audit"]:
            self.assertFalse(set(fold["fit_ids"]) & set(fold["heldout_ids"]))
            ids.extend(fold["heldout_ids"])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(self.splits["train"].dropna(subset=["market"]).row_id))

    def test_causal_estimator_recovers_known_synthetic_effect(self):
        data = make_data(Config(seed=101, days=300))
        result = partialling_out(data, 101)
        self.assertAlmostEqual(result["estimate"], -1.3, delta=.1)
        self.assertGreater(abs(result["naive_unadjusted"] + 1.3), abs(result["estimate"] + 1.3))

    def test_causal_overlap_failure_is_explicit(self):
        with self.assertRaises(ValueError):
            partialling_out(self.data.assign(price=100.0), 18)

    def test_training_uses_validation_and_is_repeatable(self):
        x = np.linspace(-1, 1, 40, dtype=np.float32)[:, None]
        y = x[:, 0] * 2
        a, record_a = train_mlp(x, y, x, y, 5, 18)
        b, record_b = train_mlp(x, y, x, y, 5, 18)
        self.assertEqual(record_a, record_b)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())))
        self.assertTrue(1 <= record_a["selected_step"] <= 5)

    def test_full_pipeline_artifacts_reconcile(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            report = run(self.config, output)
            recorded = json.loads((output / "report.json").read_text())
            self.assertEqual(recorded["storage"], report["storage"])
            rows = pd.read_csv(output / "predictions.csv")
            self.assertEqual(len(rows), report["storage"]["prediction_rows"])
            engine = create_engine("sqlite:///" + (output / "results.sqlite").as_posix())
            try:
                with engine.connect() as connection:
                    self.assertEqual(connection.scalar(text("SELECT COUNT(*) FROM predictions")), len(rows))
            finally:
                engine.dispose()
            self.assertIn("plotly", (output / "dashboard.html").read_text().lower())
            clean = rows.loc[rows.evaluation_split == "test"]
            self.assertAlmostEqual(regression_metrics(clean.demand.to_numpy(), clean.mlp_log_target.to_numpy())["rmse"],
                                   report["evaluation"]["test"]["regression"]["mlp_log_target"]["rmse"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
