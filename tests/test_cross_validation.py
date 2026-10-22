"""Tests for grouped CV isolation, OOF coverage and aggregate correctness."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_cross_validation import (CV_ROOT, IndexedFeatures, MODEL_NAMES, METRIC_NAMES,
                                 load_development, make_splits, summarize_folds)
from run_experiment import metrics, choose_ensemble


class SplitTests(unittest.TestCase):
    def test_outer_coverage_once_inner_isolation_and_test_separation(self):
        development, gold, removed = load_development()
        self.assertEqual((len(development), len(gold), len(removed)), (3601, 1120, 1))
        seen = []
        self.assertFalse(set(development.group) & set(gold.group))
        for fold, fit, val, evaluation in make_splits(development):
            for left, right in [(fit, val), (fit, evaluation), (val, evaluation)]:
                self.assertFalse(set(development.iloc[left].group) & set(development.iloc[right].group))
            self.assertEqual(set(fit) | set(val) | set(evaluation), set(range(len(development))))
            seen.extend(evaluation)
        self.assertEqual(sorted(seen), list(range(len(development))))

    def test_indexed_features_preserve_order_without_dense_copy(self):
        source = np.arange(10*5*3).reshape(10, 5, 3)
        view = IndexedFeatures(source, [7, 1, 9])
        self.assertEqual(len(view), 3)
        np.testing.assert_equal(view[0], source[7])
        self.assertTrue(np.shares_memory(view[1], source))


@unittest.skipUnless((CV_ROOT / "results/completion.json").exists(), "Full CV is not finished")
class CompletedCVTests(unittest.TestCase):
    def test_all_folds_completed_and_no_augmentation_leakage(self):
        for fold in range(1, 6):
            folder = CV_ROOT / f"fold_{fold:02d}"
            info = json.loads((folder / "results/complete.json").read_text())
            self.assertEqual(info["models"], MODEL_NAMES)
            fit = pd.read_csv(folder / "data/inner_fit.csv")
            inner = pd.read_csv(folder / "data/inner_validation.csv")
            outer = pd.read_csv(folder / "data/outer_evaluation.csv")
            augmented = pd.read_csv(folder / "data/train_augmented.csv")
            gold = pd.read_csv(CV_ROOT / "data/test.csv")
            self.assertTrue(set(augmented.parent_id).issubset(fit.row_id))
            self.assertFalse(set(augmented.group) & (set(inner.group) | set(outer.group) | set(gold.group)))
            labels = fit.set_index("row_id").label.to_dict()
            for row in augmented.itertuples():
                self.assertEqual(row.text[row.start:row.end], row.aspect)
                self.assertEqual(row.label, labels[row.parent_id])

    def test_checkpoint_and_ensemble_selection_use_inner_validation(self):
        for fold in range(1, 6):
            folder = CV_ROOT / f"fold_{fold:02d}"
            for path in (folder / "results").glob("history_*.csv"):
                history = pd.read_csv(path)
                name = path.stem.removeprefix("history_")
                checkpoint = torch.load(folder / "models" / f"{name}.pt", map_location="cpu", weights_only=True)
                self.assertAlmostEqual(checkpoint["val_f1_macro"], history.val_f1_macro.max(), places=6)
                self.assertEqual(checkpoint["best_epoch"], int(history.loc[history.val_f1_macro.idxmax(), "epoch"]))
            stored = json.loads((folder / "models/ensemble.json").read_text())
            inner_predictions = {}
            for name in stored["members"]:
                frame = pd.read_csv(folder / "results" / f"predictions_inner_validation_{name}.csv")
                inner_predictions[name] = frame[["p_negative", "p_neutral", "p_positive"]].values
            y = frame.label.map({"negative": 0, "neutral": 1, "positive": 2})
            recalculated = choose_ensemble(inner_predictions, y)
            np.testing.assert_allclose(stored["weights"], recalculated["weights"])

    def test_metrics_each_outer_fold_and_pooled_oof(self):
        all_folds = pd.read_csv(CV_ROOT / "results/metrics_folds.csv")
        self.assertEqual(len(all_folds), 50)
        assignments = pd.read_csv(CV_ROOT / "data/fold_assignments.csv")
        oof_metrics = pd.read_csv(CV_ROOT / "results/metrics_oof.csv").set_index("model")
        for name in MODEL_NAMES:
            pooled = pd.read_csv(CV_ROOT / "results" / f"predictions_oof_{name}.csv")
            self.assertEqual(pooled.row_id.tolist(), assignments.row_id.tolist())
            self.assertEqual(pooled.row_id.nunique(), 3601)
            pooled_y = pooled.label.map({"negative": 0, "neutral": 1, "positive": 2})
            pooled_values = metrics(pooled_y, pooled[["p_negative", "p_neutral", "p_positive"]].values)
            for key, value in pooled_values.items():
                self.assertAlmostEqual(oof_metrics.loc[name, key], value, places=6)
            for fold in range(1, 6):
                part = pd.read_csv(CV_ROOT / f"fold_{fold:02d}/results/predictions_outer_{name}.csv")
                expected = assignments[assignments.outer_fold == fold].row_id.tolist()
                self.assertEqual(part.row_id.tolist(), expected)
                y = part.label.map({"negative": 0, "neutral": 1, "positive": 2})
                cols = ["p_negative", "p_neutral", "p_positive"]
                values = metrics(y, part[cols].values)
                stored = all_folds[(all_folds.model == name) & (all_folds.fold == fold)].iloc[0]
                for key, value in values.items():
                    self.assertAlmostEqual(stored[key], value, places=6)
                np.testing.assert_allclose(pooled.set_index("row_id").loc[expected, cols].values,
                                           part[cols].values, atol=1e-7)

    def test_summary_sample_sd_and_cv_based_selection(self):
        folds = pd.read_csv(CV_ROOT / "results/metrics_folds.csv")
        stored = pd.read_csv(CV_ROOT / "results/metrics_cv_summary.csv")
        recalculated = summarize_folds(folds)
        pd.testing.assert_frame_equal(stored, recalculated, check_exact=False, atol=1e-12)
        selected = json.loads((CV_ROOT / "models/selected_model.json").read_text())
        self.assertEqual(selected["model"], stored.iloc[0].model)
        self.assertEqual(selected["std_ddof"], 1)

    def test_official_test_prediction_is_average_of_five_fold_models(self):
        metrics_table = pd.read_csv(CV_ROOT / "results/metrics_test.csv").set_index("model")
        gold = pd.read_csv(CV_ROOT / "data/test.csv")
        for name in MODEL_NAMES:
            probabilities = []
            for fold in range(1, 6):
                with np.load(CV_ROOT / f"fold_{fold:02d}/results/official_test_probabilities.npz") as data:
                    probabilities.append(data[name])
            expected = np.mean(np.stack(probabilities).astype(np.float64), axis=0)
            frame = pd.read_csv(CV_ROOT / "results" / f"predictions_test_{name}.csv")
            self.assertEqual(frame[["text", "aspect", "label"]].values.tolist(),
                             gold[["text", "aspect", "label"]].values.tolist())
            np.testing.assert_allclose(frame[["p_negative", "p_neutral", "p_positive"]], expected, atol=1e-7)
            recomputed = metrics(gold.y, expected)
            for key, value in recomputed.items():
                self.assertAlmostEqual(metrics_table.loc[name, key], value, places=6)


if __name__ == "__main__":
    unittest.main()
