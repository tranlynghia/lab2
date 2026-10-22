"""Full grouped 5-fold evaluation with an inner holdout for early stopping.

Every development example is evaluated exactly once in an OUTER fold. Neither
early stopping nor ensemble weights see its labels in that fold. The official
test set remains separate and is evaluated after CV-based model selection.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch

import run_experiment as core
from data_utils import LABELS, augment_train, baseline_text, read_aspect_csv
from networks import SentimentNet

ROOT = Path(__file__).resolve().parent
CV_ROOT = ROOT / "cross_validation"
EXPERIMENTS = [
    ("Random_Hybrid", "hybrid", False, False),
    ("BERT_CNN", "cnn", True, False),
    ("BERT_BiLSTM_Attention", "bilstm", True, False),
    ("BERT_Hybrid", "hybrid", True, False),
    ("BERT_Hybrid_Augmented", "hybrid", True, True),
]
BASELINES = ["Majority", "TFIDF_NaiveBayes", "TFIDF_LogisticRegression", "TFIDF_LinearSVM"]
MODEL_NAMES = BASELINES + [e[0] for e in EXPERIMENTS] + ["SoftVoting_Ensemble"]
METRIC_NAMES = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_ovr_macro"]


class IndexedFeatures:
    """View arbitrary rows of a memory map without copying all BERT features."""
    def __init__(self, source, indices):
        self.source = source
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]


def load_development():
    original = read_aspect_csv(ROOT / "data/restaurant_aspect_train.csv")
    gold = read_aspect_csv(ROOT / "data/restaurant_aspect_test.csv")
    removed = original[original.group.isin(set(gold.group))]
    development = original[~original.group.isin(set(gold.group))].reset_index(drop=True)
    return development, gold, removed


def make_splits(development, seed=42):
    """Yield disjoint inner-fit, inner-validation, and outer-evaluation indices."""
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (outer_train, outer_eval) in enumerate(
            outer.split(development, development.y, development.group), start=1):
        candidates = development.iloc[outer_train]
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed+fold)
        fit_relative, val_relative = next(inner.split(candidates, candidates.y, candidates.group))
        fit, val = outer_train[fit_relative], outer_train[val_relative]
        for a, b in [(fit, val), (fit, outer_eval), (val, outer_eval)]:
            assert set(development.iloc[a].group).isdisjoint(development.iloc[b].group)
        for index in [fit, val, outer_eval]:
            assert set(development.iloc[index].y) == {0, 1, 2}
        yield fold, fit, val, outer_eval


def release_features(arrays, cache_dir, keep_cache):
    # Close Windows memory maps before removing ONLY generated .npy files here.
    for value in arrays.values():
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()
    arrays.clear()
    gc.collect()
    cache_dir = Path(cache_dir).resolve()
    assert cache_dir.is_relative_to(CV_ROOT.resolve())
    if not keep_cache:
        for path in cache_dir.glob("bert_*.npy"):
            assert path.resolve().parent == cache_dir
            path.unlink()


def prediction_from_checkpoint(checkpoint, tokens, features, frame, config, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = SentimentNet(**saved["model_kwargs"]).to(device).eval()
    model.load_state_dict(saved["state_dict"])
    inputs = features if saved["model_kwargs"]["pretrained"] else tokens["input_ids"]
    loader = core.make_loader(inputs, tokens, frame.y, config)
    result = core.probabilities(model, loader, device)
    del model, saved
    return result


def summarize_folds(fold_metrics):
    rows = []
    for name in MODEL_NAMES:
        group = fold_metrics[fold_metrics.model == name]
        assert len(group) == 5 and group.fold.nunique() == 5
        row = dict(model=name, folds=5)
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)


def run_cv(config=None, keep_cache=False):
    config = config or core.Config()
    if config.epochs < 1 or config.patience < 1:
        raise ValueError("epochs and patience must be positive")
    started = time.perf_counter()
    for folder in ["results", "models", "data", "cache"]:
        (CV_ROOT / folder).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    core.seed_everything(config.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device)
    development, gold, removed = load_development()
    development.to_csv(CV_ROOT / "data/development.csv", index=False)
    gold.to_csv(CV_ROOT / "data/test.csv", index=False)
    removed.to_csv(CV_ROOT / "data/removed_train_overlap.csv", index=False)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / "data").glob("restaurant_aspect_*.csv")}
    code_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in ["run_cross_validation.py", "run_experiment.py", "data_utils.py", "networks.py"]}
    signature = dict(config=asdict(config), source_sha256=source_hashes, code_sha256=code_hashes,
                     protocol="5 outer grouped folds; one inner grouped holdout; no outer-label tuning")
    config_path = CV_ROOT / "results/run_signature.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != signature:
        raise RuntimeError("CV directory belongs to another configuration/source. Archive it before a new run.")
    core.dump_json(config_path, signature)
    core.dump_json(CV_ROOT / "results/config.json", asdict(config))
    splits = list(make_splits(development, config.seed))
    assignments = development[["row_id", "sentence_id", "text", "aspect", "label", "group"]].copy()
    assignments["outer_fold"] = 0
    split_info = []
    for fold, fit, val, evaluation in splits:
        assignments.loc[evaluation, "outer_fold"] = fold
        split_info.append(dict(fold=fold, inner_fit=len(fit), inner_validation=len(val), outer_evaluation=len(evaluation)))
    assert (assignments.outer_fold > 0).all()
    assignments.to_csv(CV_ROOT / "data/fold_assignments.csv", index=False)
    audit = dict(source_sha256=source_hashes, original_train=len(development)+len(removed),
                 removed_overlap=len(removed), development=len(development), test=len(gold),
                 labels=LABELS, folds=split_info,
                 selection="highest mean outer-fold macro F1; test is not a selection input",
                 early_stopping="inner validation only; no refit on all outer training rows",
                 final_test="equal probability average of five fold estimators for each method")
    core.dump_json(CV_ROOT / "results/data_audit.json", audit)
    print(f"FULL 5-FOLD CV: device={device}; development={len(development)}; untouched test={len(gold)}", flush=True)
    # Fixed, pretrained, stateless encoder: sharing its features across folds does
    # not fit vocabulary, parameters or statistics on evaluation examples.
    common_tokens, common_features, metadata = core.extract_features(
        dict(development=development, official_test=gold), config, device, CV_ROOT / "cache")
    core.dump_json(CV_ROOT / "results/bert_features.json", metadata)
    outer_prob = {name: np.full((len(development), 3), np.nan, dtype=np.float32) for name in MODEL_NAMES}
    seen = np.zeros(len(development), dtype=int)
    fold_tables = []
    for fold, fit_idx, val_idx, eval_idx in splits:
        fold_root = CV_ROOT / f"fold_{fold:02d}"
        for folder in ["models", "results", "data", "cache"]:
            (fold_root / folder).mkdir(parents=True, exist_ok=True)
        complete_path = fold_root / "results/complete.json"
        if complete_path.exists():
            print(f"Fold {fold}/5: resume previously completed fold", flush=True)
            with np.load(fold_root / "results/outer_probabilities.npz") as predictions:
                assert np.array_equal(predictions["row_indices"], eval_idx)
                for name in MODEL_NAMES:
                    outer_prob[name][eval_idx] = predictions[name]
            table = pd.read_csv(fold_root / "results/metrics_outer.csv")
        else:
            fold_start = time.perf_counter()
            fit = development.iloc[fit_idx].reset_index(drop=True)
            inner_val = development.iloc[val_idx].reset_index(drop=True)
            outer_eval = development.iloc[eval_idx].reset_index(drop=True)
            forbidden = set(inner_val.group) | set(outer_eval.group) | set(gold.group)
            augmented = augment_train(fit, forbidden, config.seed)
            for name, frame in [("inner_fit", fit), ("inner_validation", inner_val),
                                ("outer_evaluation", outer_eval), ("train_augmented", augmented)]:
                frame.to_csv(fold_root / "data" / f"{name}.csv", index=False)
            print(f"\nFold {fold}/5: fit={len(fit)} inner_val={len(inner_val)} outer_eval={len(outer_eval)} augmented={len(augmented)}", flush=True)
            ivp, ep, times = core.train_baselines(fit, inner_val, outer_eval, config, fold_root)
            aug_tokens, aug_features, aug_meta = core.extract_features(
                dict(augmented=augmented), config, device, fold_root / "cache")
            core.dump_json(fold_root / "results/augmentation.json", dict(
                counts=augmented.augmentation.value_counts().to_dict(), features=aug_meta))
            frames = dict(train=fit, validation=inner_val, test=outer_eval, augmented=augmented)
            tokens, features = {}, {}
            for key, indices in [("train", fit_idx), ("validation", val_idx), ("test", eval_idx)]:
                tokens[key] = {k: v[indices] for k, v in common_tokens["development"].items()}
                features[key] = IndexedFeatures(common_features["development"], indices)
            tokens["augmented"], features["augmented"] = aug_tokens["augmented"], aug_features["augmented"]
            for name, architecture, pretrained, aug in EXPERIMENTS:
                ivp[name], ep[name], times[name] = core.train_neural(
                    name, architecture, pretrained, aug, frames, tokens, features,
                    config, device, metadata, fold_root)
            ensemble = core.choose_ensemble(ivp, inner_val.y)
            core.dump_json(fold_root / "models/ensemble.json", ensemble)
            for prediction_sets in [ivp, ep]:
                prediction_sets["SoftVoting_Ensemble"] = sum(
                    w*prediction_sets[name] for name, w in zip(ensemble["members"], ensemble["weights"]))
            core.save_evaluation(inner_val, ivp, "inner_validation", fold_root)
            table = core.save_evaluation(outer_eval, ep, "outer", fold_root)
            np.savez_compressed(fold_root / "results/outer_probabilities.npz", row_indices=eval_idx, **ep)
            for name in MODEL_NAMES:
                outer_prob[name][eval_idx] = ep[name]
            del features, frames, tokens, aug_tokens
            release_features(aug_features, fold_root / "cache", keep_cache)
            complete = dict(fold=fold, inner_fit=len(fit), inner_validation=len(inner_val),
                            outer_evaluation=len(outer_eval), seconds=time.perf_counter()-fold_start,
                            model_seconds=times, models=MODEL_NAMES)
            core.dump_json(complete_path, complete)
            print(f"Fold {fold}/5 COMPLETE in {complete['seconds']:.1f}s. Outer scores:\n"
                  + table[["model", "accuracy", "f1_macro"]].to_string(index=False), flush=True)
        seen[eval_idx] += 1
        fold_tables.append(table.assign(fold=fold))
    assert np.all(seen == 1), "Every development example must receive one outer prediction"
    for array in outer_prob.values():
        assert np.isfinite(array).all()
    fold_metrics = pd.concat(fold_tables, ignore_index=True)
    fold_metrics.to_csv(CV_ROOT / "results/metrics_folds.csv", index=False)
    summary = summarize_folds(fold_metrics)
    summary.to_csv(CV_ROOT / "results/metrics_cv_summary.csv", index=False)
    core.save_evaluation(development, outer_prob, "oof", CV_ROOT)
    selected = summary.iloc[0].model
    selection = dict(model=selected, criterion="mean outer-fold macro F1", folds=5,
                     f1_macro_mean=float(summary.iloc[0].f1_macro_mean),
                     f1_macro_std=float(summary.iloc[0].f1_macro_std),
                     std_ddof=1, inference="equal average of five fold estimators",
                     fold_paths=[f"fold_{fold:02d}" for fold in range(1, 6)])
    core.dump_json(CV_ROOT / "models/selected_model.json", selection)
    print(f"CV SELECTION FROZEN BEFORE TEST: {selected}", flush=True)
    # Only now evaluate the untouched official test set. A mean of five test
    # probabilities is a fold ensemble, NOT five independent test folds.
    test_predictions = {name: np.zeros((len(gold), 3), dtype=np.float64) for name in MODEL_NAMES}
    test_fold_tables = []
    for fold in range(1, 6):
        fold_root = CV_ROOT / f"fold_{fold:02d}"
        predictions = {}
        for name in BASELINES:
            model = joblib.load(fold_root / "models" / f"{name}.joblib")
            assert list(model.classes_) == [0, 1, 2]
            predictions[name] = model.predict_proba(baseline_text(gold))
        for name, *_ in EXPERIMENTS:
            predictions[name] = prediction_from_checkpoint(
                fold_root / "models" / f"{name}.pt", common_tokens["official_test"],
                common_features["official_test"], gold, config, device)
        ensemble = json.loads((fold_root / "models/ensemble.json").read_text())
        predictions["SoftVoting_Ensemble"] = sum(
            w*predictions[name] for name, w in zip(ensemble["members"], ensemble["weights"]))
        np.savez_compressed(fold_root / "results/official_test_probabilities.npz", **predictions)
        for name in MODEL_NAMES:
            test_predictions[name] += predictions[name]/5
        test_fold_tables.append(pd.DataFrame([dict(fold=fold, model=name, **core.metrics(gold.y, p))
                                             for name, p in predictions.items()]))
    pd.concat(test_fold_tables, ignore_index=True).to_csv(CV_ROOT / "results/metrics_same_test_by_fold.csv", index=False)
    test_table = core.save_evaluation(gold, test_predictions, "test", CV_ROOT)
    release_features(common_features, CV_ROOT / "cache", keep_cache)
    elapsed = time.perf_counter()-started
    core.dump_json(CV_ROOT / "results/environment.json", dict(
        python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
        pandas=pd.__version__, sklearn=core.sklearn.__version__, transformers=core.transformers.__version__,
        device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None))
    core.dump_json(CV_ROOT / "results/completion.json", dict(
        completed_folds=5, methods_per_fold=10, total_outer_evaluations=50,
        development_rows=len(development), oof_coverage="exactly once per development row",
        elapsed_seconds_this_invocation=elapsed, selected=selected))
    from report_cv import write_cv_report
    write_cv_report()
    print(summary[["model", "f1_macro_mean", "f1_macro_std"]].to_string(index=False), flush=True)
    print(test_table.to_string(index=False), flush=True)
    print(f"FULL CV COMPLETE: {elapsed:.1f}s", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    # Write UTF-8 logs directly; avoid PowerShell's native stderr redirection.
    (CV_ROOT / "results").mkdir(parents=True, exist_ok=True)
    class Tee:
        def __init__(self, terminal, log):
            self.terminal, self.log = terminal, log
        def write(self, value):
            self.terminal.write(value)
            self.log.write(value)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        def isatty(self):
            return False
    with (CV_ROOT / "results/run.log").open("a", encoding="utf-8") as log:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
        try:
            run_cv(core.Config(epochs=args.epochs, patience=args.patience, seed=args.seed,
                               device=args.device, local_files_only=not args.allow_download), args.keep_cache)
        except Exception:
            import traceback
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
