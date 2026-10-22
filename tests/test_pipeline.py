import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import read_aspect_csv, augment_train, normalize
from networks import SentimentNet
from run_experiment import metrics, choose_ensemble


class PipelineTests(unittest.TestCase):
    def test_source_csv_is_unchanged(self):
        for name in ["restaurant_aspect_train.csv", "restaurant_aspect_test.csv"]:
            source = ROOT.parent / "data" / name
            if source.exists():
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                                 hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest())
        self.assertEqual(len(read_aspect_csv(ROOT / "data/restaurant_aspect_test.csv")), 1120)

    def test_disjoint_sentence_groups_and_unchanged_gold_test(self):
        frames = [pd.read_csv(ROOT / "data" / f"{name}.csv") for name in ["train", "validation", "test"]]
        for i, a in enumerate(frames):
            for b in frames[i+1:]:
                self.assertFalse(set(a.group) & set(b.group))
        source = read_aspect_csv(ROOT / "data/restaurant_aspect_test.csv")
        self.assertEqual(frames[2][["text", "aspect", "label"]].values.tolist(),
                         source[["text", "aspect", "label"]].values.tolist())

    def test_augmentation_protects_target_negation_and_offsets(self):
        text = "The very friendly staff were not really good."
        start = text.index("staff")
        row = dict(row_id="x:0", sentence_id="x", text=text, aspect="staff", start=start,
                   end=start+5, label="negative", y=0, group=normalize(text))
        result = augment_train(pd.DataFrame([row]), seed=42)
        self.assertEqual(set(result.augmentation), {"original", "synonym", "restricted_deletion"})
        for record in result.to_dict("records"):
            self.assertIn("not", record["text"].split())
            self.assertEqual(record["text"][record["start"]:record["end"]], "staff")
            self.assertEqual(record["y"], 0)
        blocked = augment_train(pd.DataFrame([row]), forbidden_groups=set(result.group), seed=42)
        self.assertEqual(len(blocked), 1)

    def test_mixed_sentiment_sentence_keeps_distinct_aspects(self):
        data = read_aspect_csv(ROOT / "data/restaurant_aspect_train.csv")
        groups = data.groupby("text").label.nunique()
        mixed = data[data.text == groups[groups > 1].index[0]]
        self.assertGreater(mixed.aspect.nunique(), 1)
        self.assertGreater(mixed.label.nunique(), 1)

    def test_padding_does_not_change_prediction(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)
        for architecture in ["cnn", "bilstm", "hybrid"]:
            for pretrained in [True, False]:
                model = SentimentNet(architecture=architecture, pretrained=pretrained,
                                     vocab_size=20, feature_dim=8, hidden=4, projection=6).eval()
                x = torch.randn(2, 5, 8) if pretrained else torch.randint(1, 20, (2, 5))
                padded = torch.cat([x, torch.zeros((2, 4, 8) if pretrained else (2, 4), dtype=x.dtype)], dim=1)
                with torch.no_grad():
                    a = model(x, torch.ones(2, 5))
                    b = model(padded, torch.cat([torch.ones(2, 5), torch.zeros(2, 4)], dim=1))
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_metrics_and_ensemble(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.eye(3)[y]*.8+.2/3
        result = metrics(y, probabilities)
        self.assertTrue(all(abs(value-1) < 1e-9 for value in result.values()))
        predictions = {name: probabilities for name in
                       ["BERT_CNN", "BERT_BiLSTM_Attention", "BERT_Hybrid_Augmented"]}
        ensemble = choose_ensemble(predictions, y)
        self.assertAlmostEqual(sum(ensemble["weights"]), 1)
        self.assertTrue(all(w > 0 for w in ensemble["weights"]))

    def test_saved_results_match_recomputed_metrics(self):
        path = ROOT / "results/metrics_test.csv"
        if not path.exists():
            self.skipTest("Training has not completed")
        table = pd.read_csv(path)
        self.assertEqual(len(table), 10)
        for row in table.itertuples():
            output = pd.read_csv(ROOT / "results" / f"predictions_test_{row.model}.csv")
            self.assertEqual(len(output), 1120)
            y = output.label.map({"negative": 0, "neutral": 1, "positive": 2})
            recomputed = metrics(y, output[["p_negative", "p_neutral", "p_positive"]].values)
            for metric, value in recomputed.items():
                self.assertAlmostEqual(getattr(row, metric), value, places=6)
        val = pd.read_csv(ROOT / "results/metrics_validation.csv")
        selected = json.loads((ROOT / "models/selected_model.json").read_text())["model"]
        self.assertEqual(selected, val.sort_values("f1_macro", ascending=False).iloc[0].model)


if __name__ == "__main__":
    unittest.main()
