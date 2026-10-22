"""SemEval aspect polarity data, leakage-safe splits, and conservative EDA."""
from pathlib import Path
import hashlib
import random
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def read_aspect_csv(path):
    """Use the user's standardized aspect CSV without relabeling any sentence."""
    frame = pd.read_csv(path, dtype={"sentence_id": str})
    required = {"sentence_id", "text", "aspect_term", "from", "to", "polarity"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing CSV columns: {required-set(frame.columns)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing values in required aspect fields")
    if not frame.polarity.isin(LABELS).all():
        raise ValueError("Expected the existing three-class aspect CSV")
    frame = frame.rename(columns={"aspect_term": "aspect", "from": "start",
                                  "to": "end", "polarity": "label"})
    for row in frame.itertuples():
        if row.text[row.start:row.end] != row.aspect:
            raise ValueError(f"Invalid target span: {row.sentence_id}")
    frame["row_id"] = [f"{sid}:{i}" for i, sid in enumerate(frame.sentence_id)]
    frame["y"] = frame.label.map(LABEL_TO_ID)
    frame["group"] = frame.text.map(normalize)
    # Derived text fields from the old lab are recomputed when needed, so an
    # augmented row never carries an out-of-date feature_text/context_window.
    return frame.drop(columns=["context_window", "feature_text"], errors="ignore")


def prepare_data(data_dir, seed=42):
    data_dir = Path(data_dir)
    train_all = read_aspect_csv(data_dir / "restaurant_aspect_train.csv")
    test = read_aspect_csv(data_dir / "restaurant_aspect_test.csv")
    original_count = len(train_all)
    # Preserve the gold test set; remove exact normalized sentence overlap from train.
    train_all = train_all[~train_all.group.isin(set(test.group))].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(train_all, train_all.y, train_all.group))
    train = train_all.iloc[train_idx].reset_index(drop=True)
    val = train_all.iloc[val_idx].reset_index(drop=True)
    for a, b in [(train, val), (train, test), (val, test)]:
        assert set(a.group).isdisjoint(b.group)
    for name, frame in [("train", train), ("validation", val), ("test", test)]:
        assert set(frame.y) == {0, 1, 2}, f"Missing class in {name}"
        frame.to_csv(data_dir / f"{name}.csv", index=False)
    audit = dict(raw_train_aspects=original_count, raw_test_aspects=len(test),
                 removed_train_overlap=original_count-len(train_all), seed=seed,
                 split_method="first fold of StratifiedGroupKFold(5), group=normalized sentence",
                 labels=LABELS, source_sha256={
                     p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in data_dir.glob("restaurant_aspect_*.csv")},
                 splits={name: dict(rows=len(df), sentences=df.group.nunique(),
                                    classes=df.label.value_counts().to_dict())
                         for name, df in [("train", train), ("validation", val), ("test", test)]})
    return train, val, test, audit


SYNONYMS = {"good": "nice", "excellent": "superb", "delicious": "tasty",
            "friendly": "welcoming", "rude": "impolite", "slow": "sluggish",
            "quick": "fast", "expensive": "costly", "lovely": "delightful",
            "awful": "terrible", "wonderful": "fantastic", "tasty": "delicious"}
DELETABLE = {"really", "very", "quite", "just"}
NEGATIONS = {"not", "no", "never", "neither", "nor", "without", "hardly"}


def augment_train(train, forbidden_groups=(), seed=42):
    """At most one synonym replacement and one restricted deletion per example.

    Protect the annotated target span and every occurrence of its words. The
    dictionary is fixed in advance and does not learn from validation/test data.
    These are heuristic label-preserving transforms, not a semantic guarantee.
    """
    rng = random.Random(seed)
    forbidden = set(forbidden_groups)
    seen = set(zip(train.group, train.aspect.str.lower(), train.y))
    generated = []
    for record in train.to_dict("records"):
        target_words = set(re.findall(r"\w+", record["aspect"].lower()))
        candidates = []
        for match in re.finditer(r"\b[a-zA-Z]+\b", record["text"]):
            word = match.group().lower()
            overlap = match.start() < record["end"] and match.end() > record["start"]
            if not overlap and word not in target_words and word not in NEGATIONS:
                candidates.append((match, word))
        for method in ["synonym", "restricted_deletion"]:
            eligible = [(m, w) for m, w in candidates
                        if w in (SYNONYMS if method == "synonym" else DELETABLE)]
            if not eligible:
                continue
            match, word = rng.choice(eligible)
            replacement = SYNONYMS[word] if method == "synonym" else ""
            text = record["text"][:match.start()] + replacement + record["text"][match.end():]
            # Keep character offsets valid, including a deletion before the target.
            shift = len(replacement) - (match.end()-match.start()) if match.end() <= record["start"] else 0
            key = (normalize(text), record["aspect"].lower(), record["y"])
            if key in seen or key[0] in forbidden:
                continue
            seen.add(key)
            item = dict(record, row_id=record["row_id"]+":"+method,
                        parent_id=record["row_id"], text=text, group=key[0],
                        start=record["start"]+shift, end=record["end"]+shift,
                        augmentation=method)
            assert text[item["start"]:item["end"]] == item["aspect"]
            generated.append(item)
    original = train.assign(parent_id=train.row_id, augmentation="original")
    return pd.concat([original, pd.DataFrame(generated)], ignore_index=True)


def baseline_text(frame):
    # Use the known aspect and local context; all labels remain excluded.
    result = []
    for row in frame.itertuples():
        left = " ".join(row.text[:row.start].split()[-5:])
        right = " ".join(row.text[row.end:].split()[:5])
        result.append(f"{row.text} ASPECT {row.aspect} {row.aspect} CONTEXT {left} {right}")
    return result
