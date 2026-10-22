"""Predict sentiment for a known aspect, using a saved neural model or ensemble."""
import os
os.environ.setdefault("USE_TF", "0")
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from networks import SentimentNet

ROOT = Path(__file__).resolve().parent


def predict(text, aspect, model_name="SoftVoting_Ensemble", allow_download=False):
    if not text.strip() or not aspect.strip():
        raise ValueError("Both text and aspect are required")
    if aspect.lower() not in text.lower():
        raise ValueError("Aspect term must occur in the review")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    if model_name == "SoftVoting_Ensemble":
        spec = json.loads((ROOT / "models/ensemble.json").read_text())
        names, weights = spec["members"], spec["weights"]
    else:
        names, weights = [model_name], [1.0]
    checkpoints = [torch.load(ROOT / "models" / f"{name}.pt", map_location="cpu", weights_only=True) for name in names]
    cfg = checkpoints[0]["config"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["bert_model"], local_files_only=not allow_download)
    tokens = tokenizer([aspect], [text], padding=True, truncation="only_second",
                       max_length=cfg["max_length"], return_tensors="pt").to(device)
    features = None
    if any(c["model_kwargs"]["pretrained"] for c in checkpoints):
        encoder = AutoModel.from_pretrained(cfg["bert_model"], local_files_only=not allow_download).to(device).eval()
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = encoder(**tokens).last_hidden_state
            else:
                features = encoder(**tokens).last_hidden_state
        features = features.half()  # Same quantization as the feature cache.
        del encoder
    probability = np.zeros(3)
    for checkpoint, weight in zip(checkpoints, weights):
        model = SentimentNet(**checkpoint["model_kwargs"]).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"])
        inputs = features if checkpoint["model_kwargs"]["pretrained"] else tokens["input_ids"]
        with torch.inference_mode():
            probability += weight * model(inputs, tokens["attention_mask"]).softmax(-1)[0].cpu().numpy()
    labels = checkpoints[0]["labels"]
    return dict(text=text, aspect=aspect, model=model_name, label=labels[int(probability.argmax())],
                probabilities=dict(zip(labels, probability.tolist())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--aspect", required=True)
    parser.add_argument("--model", default="SoftVoting_Ensemble")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(predict(args.text, args.aspect, args.model, args.allow_download), ensure_ascii=False, indent=2))
