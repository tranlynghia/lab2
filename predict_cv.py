"""Predict with the method selected by full CV, averaging its five fold models."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from run_experiment import Config
from run_cross_validation import BASELINES, CV_ROOT, prediction_from_checkpoint
from data_utils import LABELS, baseline_text


def predict_batch(texts, aspects, allow_download=False):
    if len(texts) != len(aspects) or not texts:
        raise ValueError("Provide equally sized nonempty lists of reviews and aspects")
    selection = json.loads((CV_ROOT / "models/selected_model.json").read_text())
    model_name = selection["model"]
    config = Config(**json.loads((CV_ROOT / "results/config.json").read_text()))
    rows = []
    for text, aspect in zip(texts, aspects):
        start = text.lower().find(aspect.lower())
        if not text.strip() or not aspect.strip() or start < 0:
            raise ValueError("Each aspect must occur in its review")
        rows.append(dict(text=text, aspect=aspect, start=start, end=start+len(aspect), y=0))
    frame = pd.DataFrame(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config.threads)
    tokens, features = None, None
    if model_name not in BASELINES:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.bert_model, local_files_only=not allow_download)
        batch = tokenizer(aspects, texts, padding=True, truncation="only_second",
                          max_length=config.max_length, return_tensors="pt").to(device)
        tokens = {k: v.cpu().numpy() for k, v in batch.items()}
        if model_name != "Random_Hybrid":
            encoder = AutoModel.from_pretrained(config.bert_model, local_files_only=not allow_download).to(device).eval()
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        features = encoder(**batch).last_hidden_state.cpu().numpy().astype("float16")
                else:
                    features = encoder(**batch).last_hidden_state.cpu().numpy().astype("float16")
            del encoder
    total = np.zeros((len(frame), 3), dtype=np.float64)
    for folder_name in selection["fold_paths"]:
        folder = CV_ROOT / folder_name
        if model_name in BASELINES:
            model = joblib.load(folder / "models" / f"{model_name}.joblib")
            probability = model.predict_proba(baseline_text(frame))
        else:
            if model_name == "SoftVoting_Ensemble":
                ensemble = json.loads((folder / "models/ensemble.json").read_text())
                names, weights = ensemble["members"], ensemble["weights"]
            else:
                names, weights = [model_name], [1.0]
            probability = sum(weight*prediction_from_checkpoint(
                folder / "models" / f"{name}.pt", tokens, features, frame, config, device)
                for name, weight in zip(names, weights))
        total += probability/len(selection["fold_paths"])
    return [dict(text=text, aspect=aspect, model=model_name, fold_models=5,
                 label=LABELS[int(p.argmax())], probabilities=dict(zip(LABELS, p.tolist())))
            for text, aspect, p in zip(texts, aspects, total)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--aspect", required=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(predict_batch([args.text], [args.aspect], args.allow_download)[0], ensure_ascii=False, indent=2))
