"""Run the complete Chapter 3 aspect-level sentiment experiment.

No fitting, early stopping, or ensemble weight selection uses test labels.
BERT is frozen; downstream neural classifiers are trained on its token features.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoModel, AutoTokenizer

from data_utils import LABELS, augment_train, baseline_text, prepare_data
from networks import SentimentNet

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    seed: int = 42
    epochs: int = 20
    patience: int = 4
    batch_size: int = 64
    feature_batch_size: int = 24
    max_length: int = 96
    learning_rate: float = 0.001
    bert_model: str = "bert-base-uncased"
    local_files_only: bool = True
    device: str = "auto"
    threads: int = 4


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    assert probabilities.shape == (len(y), len(LABELS))
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    assert np.allclose(probabilities.sum(1), 1, atol=1e-5)
    prediction = probabilities.argmax(1)
    return dict(accuracy=accuracy_score(y, prediction),
                precision_macro=precision_score(y, prediction, average="macro", zero_division=0),
                recall_macro=recall_score(y, prediction, average="macro", zero_division=0),
                f1_macro=f1_score(y, prediction, average="macro", zero_division=0),
                f1_weighted=f1_score(y, prediction, average="weighted", zero_division=0),
                roc_auc_ovr_macro=roc_auc_score(y, probabilities/probabilities.sum(1, keepdims=True),
                                              labels=[0, 1, 2], multi_class="ovr", average="macro"))


def train_baselines(train, val, test, config, output_root=None):
    output_root = Path(output_root) if output_root is not None else ROOT
    def vectorizer():
        return TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                               min_df=2, sublinear_tf=True, max_features=30000)
    cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=config.seed)
              .split(train, train.y, train.group))
    estimators = {
        "Majority": make_pipeline(vectorizer(), DummyClassifier(strategy="prior")),
        "TFIDF_NaiveBayes": make_pipeline(vectorizer(), MultinomialNB(alpha=1.0)),
        "TFIDF_LogisticRegression": make_pipeline(vectorizer(), LogisticRegression(
            C=2.0, max_iter=1500, class_weight="balanced", random_state=config.seed)),
        "TFIDF_LinearSVM": CalibratedClassifierCV(
            make_pipeline(vectorizer(), LinearSVC(C=1.0, class_weight="balanced",
                                                  random_state=config.seed)), cv=cv),
    }
    xtrain, xval, xtest = [baseline_text(df) for df in (train, val, test)]
    vp, tp, durations = {}, {}, {}
    for name, estimator in estimators.items():
        start = time.perf_counter()
        estimator.fit(xtrain, train.y)
        vp[name] = estimator.predict_proba(xval)
        tp[name] = estimator.predict_proba(xtest)
        assert list(estimator.classes_) == [0, 1, 2]
        durations[name] = time.perf_counter()-start
        joblib.dump(estimator, output_root / "models" / f"{name}.joblib")
        print(f"{name}: validation macro F1={metrics(val.y, vp[name])['f1_macro']:.4f}", flush=True)
    return vp, tp, durations


def tokenize_frame(tokenizer, frame, max_length):
    # Aspect FIRST: the target always survives truncation of long review text.
    return tokenizer(frame.aspect.tolist(), frame.text.tolist(), padding="max_length",
                     truncation="only_second", max_length=max_length, return_tensors="np")


def extract_features(frames, config, device, cache_dir=None):
    cache = Path(cache_dir) if cache_dir is not None else ROOT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, local_files_only=config.local_files_only)
    encoder = AutoModel.from_pretrained(config.bert_model,
                                        local_files_only=config.local_files_only).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    revision = getattr(encoder.config, "_commit_hash", None)
    token_sets, feature_sets, audit = {}, {}, {}
    for name, frame in frames.items():
        start = time.perf_counter()
        tokens = tokenize_frame(tokenizer, frame, config.max_length)
        token_sets[name] = tokens
        fingerprint = hashlib.sha256(json.dumps(dict(
            model=config.bert_model, revision=revision, max_length=config.max_length,
            transformers=transformers.__version__,
            data=frame[["aspect", "text"]].values.tolist()), ensure_ascii=False).encode()).hexdigest()[:20]
        path = cache / f"bert_{name}_{fingerprint}.npy"
        shape = (len(frame), config.max_length, encoder.config.hidden_size)
        if not path.exists():
            temporary = path.with_suffix(".partial.npy")
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype="float16", shape=shape)
            with torch.inference_mode():
                for offset in range(0, len(frame), config.feature_batch_size):
                    end = min(len(frame), offset+config.feature_batch_size)
                    inputs = {k: torch.as_tensor(v[offset:end], device=device) for k, v in tokens.items()}
                    # Dynamic batch trimming avoids BERT computation on trailing padding.
                    length = int(inputs["attention_mask"].sum(1).max())
                    inputs = {k: v[:, :length] for k, v in inputs.items()}
                    context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else __import__("contextlib").nullcontext()
                    with context:
                        hidden = encoder(**inputs).last_hidden_state
                    array[offset:end] = 0
                    array[offset:end, :length] = hidden.cpu().numpy().astype("float16")
                    if offset % (config.feature_batch_size*40) == 0:
                        print(f"BERT {name}: {end}/{len(frame)}", flush=True)
            array.flush()
            del array
            temporary.replace(path)
        feature_sets[name] = np.load(path, mmap_mode="r")
        assert feature_sets[name].shape == shape
        # Count actual truncated pairs, not merely full-length padded arrays.
        lengths = [len(tokenizer(a, t, truncation=False)["input_ids"])
                   for a, t in zip(frame.aspect, frame.text)]
        audit[name] = dict(rows=len(frame), truncated=sum(n > config.max_length for n in lengths),
                           seconds=time.perf_counter()-start, fingerprint=fingerprint)
    metadata = dict(model=config.bert_model, revision=revision, frozen=True,
                    hidden_size=encoder.config.hidden_size, vocab_size=len(tokenizer),
                    pair_format="[CLS] aspect [SEP] review [SEP]", splits=audit)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return token_sets, feature_sets, metadata


class FeatureDataset(Dataset):
    def __init__(self, x, mask, y):
        self.x, self.mask, self.y = x, mask, np.asarray(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(np.array(self.x[i], copy=True)),
                torch.from_numpy(np.array(self.mask[i], copy=True)),
                torch.tensor(self.y[i], dtype=torch.long))


def make_loader(x, tokens, y, config, shuffle=False):
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(FeatureDataset(x, tokens["attention_mask"], y), batch_size=config.batch_size,
                      shuffle=shuffle, num_workers=0, generator=generator)


def probabilities(model, loader, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for x, mask, _ in loader:
            predictions.append(model(x.to(device), mask.to(device)).softmax(-1).cpu().numpy())
    return np.concatenate(predictions)


def train_neural(name, architecture, pretrained, augmented, frames, tokens, features,
                 config, device, metadata, output_root=None):
    output_root = Path(output_root) if output_root is not None else ROOT
    seed_everything(config.seed)
    split = "augmented" if augmented else "train"
    def loader(key, shuffle=False):
        x = features[key] if pretrained else tokens[key]["input_ids"]
        return make_loader(x, tokens[key], frames[key].y, config, shuffle)
    train_loader, val_loader, test_loader = loader(split, True), loader("validation"), loader("test")
    model_kwargs = dict(architecture=architecture, pretrained=pretrained,
                        vocab_size=metadata["vocab_size"], feature_dim=metadata["hidden_size"])
    model = SentimentNet(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    # Same class weighting from the ORIGINAL train partition in every neural run.
    counts = np.bincount(frames["train"].y, minlength=3)
    weights = len(frames["train"])/(3.0*counts)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    history, best_f1, stale = [], -1.0, 0
    checkpoint = output_root / "models" / f"{name}.pt"
    start = time.perf_counter()
    for epoch in range(1, config.epochs+1):
        model.train()
        total, correct, seen = 0.0, 0, 0
        for x, mask, target in train_loader:
            x, mask, target = x.to(device), mask.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = loss_fn(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()*len(target)
            correct += (logits.argmax(-1) == target).sum().item()
            seen += len(target)
        val_prob = probabilities(model, val_loader, device)
        val_metrics = metrics(frames["validation"].y, val_prob)
        history.append(dict(epoch=epoch, train_loss=total/seen, train_accuracy=correct/seen,
                            val_accuracy=val_metrics["accuracy"], val_f1_macro=val_metrics["f1_macro"]))
        print(f"{name} epoch {epoch:02d}: loss={total/seen:.4f}, val F1={val_metrics['f1_macro']:.4f}", flush=True)
        if val_metrics["f1_macro"] > best_f1+1e-6:
            best_f1, stale = val_metrics["f1_macro"], 0
            torch.save(dict(state_dict=model.state_dict(), model_kwargs=model_kwargs,
                            config=asdict(config), best_epoch=epoch, val_f1_macro=best_f1,
                            labels=LABELS, bert_metadata=metadata), checkpoint)
        else:
            stale += 1
            if stale >= config.patience:
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    vp, tp = probabilities(model, val_loader, device), probabilities(model, test_loader, device)
    pd.DataFrame(history).to_csv(output_root / "results" / f"history_{name}.csv", index=False)
    elapsed = time.perf_counter()-start
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return vp, tp, elapsed


def choose_ensemble(validation_predictions, y):
    members = ["BERT_CNN", "BERT_BiLSTM_Attention", "BERT_Hybrid_Augmented"]
    stacked = np.stack([validation_predictions[n] for n in members])
    best_score, best_weights = -1.0, None
    # 27 predeclared positive-weight candidates, selected ONLY by validation F1.
    for candidate in itertools.product((1, 2, 3), repeat=3):
        weights = np.array(candidate, dtype=float)/sum(candidate)
        score = f1_score(y, np.einsum("m,mnc->nc", weights, stacked).argmax(-1), average="macro")
        if score > best_score+1e-8:
            best_score, best_weights = score, weights
    return dict(members=members, weights=best_weights.tolist(), validation_f1_macro=best_score,
                selection="27 fixed candidates; validation macro F1 only")


def save_evaluation(frame, prediction_sets, split, output_root=None):
    output_root = Path(output_root) if output_root is not None else ROOT
    rows = []
    for name, probs in prediction_sets.items():
        rows.append(dict(model=name, **metrics(frame.y, probs)))
        output = frame[["row_id", "sentence_id", "text", "aspect", "label"]].copy()
        output["predicted"] = np.array(LABELS)[probs.argmax(-1)]
        for i, label in enumerate(LABELS):
            output[f"p_{label}"] = probs[:, i]
        output.to_csv(output_root / "results" / f"predictions_{split}_{name}.csv", index=False)
        if split == "test":
            output[output.label != output.predicted].to_csv(
                output_root / "results" / f"errors_{name}.csv", index=False)
            dump_json(output_root / "results" / f"classification_report_{name}.json",
                      classification_report(frame.y, probs.argmax(-1), labels=[0, 1, 2],
                                            target_names=LABELS, output_dict=True, zero_division=0))
    table = pd.DataFrame(rows)
    table.to_csv(output_root / "results" / f"metrics_{split}.csv", index=False)
    return table


def draw_figures(test, prediction_sets, test_table, val_table):
    fig, ax = plt.subplots(figsize=(11, 6))
    order = val_table.sort_values("f1_macro", ascending=True).model.tolist()
    comparison = test_table.set_index("model").loc[order]
    y = np.arange(len(order))
    ax.barh(y-.18, comparison.accuracy, .35, label="Test accuracy", color="#009b8e")
    ax.barh(y+.18, comparison.f1_macro, .35, label="Test macro F1", color="#3455a4")
    ax.set(yticks=y, yticklabels=order, xlim=(0, 1), xlabel="Score",
           title="Aspect sentiment: identical test set (ordered by validation F1)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "model_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, (name, probs) in zip(axes.flat, prediction_sets.items()):
        cm = confusion_matrix(test.y, probs.argmax(-1), labels=[0, 1, 2])
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
               title=name.replace("_", "\n"), xlabel="Predicted", ylabel="True")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "confusion_matrices.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label_id, ax in enumerate(axes):
        for name, probs in prediction_sets.items():
            fpr, tpr, _ = roc_curve((test.y == label_id).astype(int), probs[:, label_id])
            ax.plot(fpr, tpr, label=name, linewidth=1.2)
        ax.plot([0, 1], [0, 1], "k--", alpha=.4)
        ax.set(title=f"{LABELS[label_id]} vs rest", xlabel="False positive rate", ylabel="True positive rate")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "roc_ovr.png", dpi=180)
    plt.close(fig)
    histories = list((ROOT / "results").glob("history_*.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for path in histories:
        history = pd.read_csv(path)
        name = path.stem.removeprefix("history_")
        axes[0].plot(history.epoch, history.train_loss, label=name)
        axes[1].plot(history.epoch, history.val_f1_macro, label=name)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Weighted cross-entropy")
    axes[1].set(title="Validation macro F1", xlabel="Epoch", ylabel="Macro F1")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "learning_curves.png", dpi=180)
    plt.close(fig)


def run(config=None):
    config = config or Config()
    if config.epochs < 1 or config.patience < 1 or config.max_length < 16:
        raise ValueError("epochs/patience must be positive and max_length >= 16")
    for folder in ["data", "models", "results", "cache"]:
        (ROOT / folder).mkdir(exist_ok=True)
    torch.set_num_threads(config.threads)
    seed_everything(config.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device)
    started = time.perf_counter()
    dump_json(ROOT / "results" / "config.json", asdict(config))
    train, val, test, audit = prepare_data(ROOT / "data", config.seed)
    augmented = augment_train(train, set(val.group) | set(test.group), config.seed)
    # Keep an auditable list of generated examples and the original parent ID.
    augmented.to_csv(ROOT / "data" / "train_augmented.csv", index=False)
    audit["augmentation"] = augmented.augmentation.value_counts().to_dict()
    dump_json(ROOT / "results" / "data_audit.json", audit)
    print(f"Device={device}; train={len(train)}, validation={len(val)}, test={len(test)}, augmented train={len(augmented)}", flush=True)
    vp, tp, times = train_baselines(train, val, test, config)
    frames = dict(train=train, validation=val, test=test, augmented=augmented)
    tokens, features, metadata = extract_features(frames, config, device)
    dump_json(ROOT / "results" / "bert_features.json", metadata)
    experiments = [
        ("Random_Hybrid", "hybrid", False, False),
        ("BERT_CNN", "cnn", True, False),
        ("BERT_BiLSTM_Attention", "bilstm", True, False),
        ("BERT_Hybrid", "hybrid", True, False),
        ("BERT_Hybrid_Augmented", "hybrid", True, True),
    ]
    for name, architecture, pretrained, aug in experiments:
        vp[name], tp[name], times[name] = train_neural(
            name, architecture, pretrained, aug, frames, tokens, features, config, device, metadata)
    ensemble = choose_ensemble(vp, val.y)
    dump_json(ROOT / "models" / "ensemble.json", ensemble)
    for predictions in [vp, tp]:
        predictions["SoftVoting_Ensemble"] = sum(w*predictions[n] for n, w in zip(ensemble["members"], ensemble["weights"]))
    validation_table = save_evaluation(val, vp, "validation")
    best = validation_table.sort_values("f1_macro", ascending=False).iloc[0].model
    # Selection is frozen before computing/reporting the test metrics.
    dump_json(ROOT / "models" / "selected_model.json", dict(model=best, criterion="validation macro F1"))
    test_table = save_evaluation(test, tp, "test")
    draw_figures(test, tp, test_table, validation_table)
    timing = dict(model_train_and_predict_seconds=times, total_seconds=time.perf_counter()-started,
                  bert_feature_seconds=sum(s["seconds"] for s in metadata["splits"].values()))
    dump_json(ROOT / "results" / "timing.json", timing)
    versions = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    pandas=pd.__version__, sklearn=sklearn.__version__, transformers=transformers.__version__,
                    device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
    dump_json(ROOT / "results" / "environment.json", versions)
    from write_report import write_report
    write_report()
    print(test_table.to_string(index=False), flush=True)
    print(f"Selected by validation: {best}; elapsed {timing['total_seconds']:.1f}s", flush=True)
    return test_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading BERT when it is not cached")
    args = parser.parse_args()
    run(Config(epochs=args.epochs, patience=args.patience, seed=args.seed,
               device=args.device, local_files_only=not args.allow_download))
