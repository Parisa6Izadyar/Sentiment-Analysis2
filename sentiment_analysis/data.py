"""Read Sentiment140 and construct disjoint text partitions."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .text import text_key

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = "training.1600000.processed.noemoticon.csv"
TEST_FILE = "testdata.manual.2009.06.14.csv"
COLUMNS = ["sentiment", "id", "date", "query", "user", "text"]


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_sentiment140(path, allow_neutral=False):
    frame = pd.read_csv(path, names=COLUMNS, header=None, encoding="latin-1",
                        usecols=["sentiment", "text"])
    labels = set(frame.sentiment.unique())
    allowed = {0, 2, 4} if allow_neutral else {0, 4}
    if not labels.issubset(allowed) or not {0, 4}.issubset(labels):
        raise ValueError(f"Expected Sentiment140 labels {sorted(allowed)}; found {sorted(labels)}")
    total = len(frame)
    frame = frame.loc[frame.sentiment.isin([0, 4])].copy()
    frame["label"] = frame.pop("sentiment").map({0: 0, 4: 1}).astype("int8")
    frame["source_row"] = frame.index
    return frame.reset_index(drop=True), {"file_rows": total, "binary_rows": len(frame), "sha256": checksum(path)}


def stratified_sample(frame, size, seed):
    if size == 0 or size == len(frame):
        return frame.copy()
    if size < 30 or size > len(frame):
        raise ValueError("sample-size must be zero (all rows) or between 30 and the source row count")
    selected, _ = train_test_split(frame, train_size=size, stratify=frame.label, random_state=seed)
    return selected.sort_values("source_row").reset_index(drop=True)


def clean_texts(frame):
    if frame.empty or not {"text", "label", "source_row"}.issubset(frame.columns):
        raise ValueError("Expected nonempty text, label and source_row columns")
    if frame.label.isna().any() or not set(frame.label.unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")
    stats = {"input_rows": len(frame)}
    frame = frame.copy()
    valid = frame.text.map(lambda value: isinstance(value, str) and bool(value.strip()))
    stats["missing_or_blank_rows"] = int((~valid).sum())
    frame = frame.loc[valid].copy()
    frame["text_key"] = frame.text.map(text_key)
    stats["empty_after_processing"] = int(frame.text_key.eq("").sum())
    frame = frame.loc[frame.text_key.ne("")].copy()
    counts = frame.groupby("text_key").label.nunique()
    conflicting = set(counts[counts > 1].index)
    stats["conflicting_text_groups"] = len(conflicting)
    stats["conflicting_rows_removed"] = int(frame.text_key.isin(conflicting).sum())
    frame = frame.loc[~frame.text_key.isin(conflicting)].copy()
    stats["duplicate_rows_removed"] = int(frame.duplicated("text_key").sum())
    frame = frame.drop_duplicates("text_key").sort_values("source_row").reset_index(drop=True)
    stats["retained_rows"] = len(frame)
    if set(frame.label.unique()) != {0, 1}:
        raise ValueError("Both sentiment classes must remain after cleaning")
    return frame, stats


@dataclass
class Partitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_texts(frame, seed=42):
    if frame.text_key.duplicated().any():
        raise ValueError("Deduplicate processed text before splitting")
    if len(frame) < 30 or frame.label.value_counts().min() < 10:
        raise ValueError("Need at least 30 cleaned rows and ten observations per class")
    development, test = train_test_split(frame, test_size=0.10, stratify=frame.label, random_state=seed)
    train, validation = train_test_split(development, test_size=1/9,
                                        stratify=development.label, random_state=seed)
    result = Partitions(train.copy(), validation.copy(), test.copy())
    assert_disjoint(result)
    return result


def assert_disjoint(parts):
    groups = [set(getattr(parts, name).text_key) for name in ["train", "validation", "test"]]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Processed text overlaps across partitions")
