"""Fit candidates on training data and select by validation macro F1."""
from dataclasses import dataclass

from .data import assert_disjoint
from .evaluate import score_model
from .model import make_classifiers, make_vectorizer


@dataclass
class FittedExperiment:
    vectorizer: object
    models: dict
    validation: dict
    selected: str


def choose_model(validation):
    candidates = sorted(name for name in validation if name != "majority_baseline")
    return max(candidates, key=lambda name: validation[name]["macro_f1"])


def fit_candidates(parts, seed=42, max_features=100000, min_df=2):
    assert_disjoint(parts)
    vectorizer = make_vectorizer(max_features=max_features, min_df=min_df)
    print(f"Fitting TF-IDF on {len(parts.train):,} training texts...", flush=True)
    train_features = vectorizer.fit_transform(parts.train.text)
    validation_features = vectorizer.transform(parts.validation.text)
    models = make_classifiers(seed)
    validation = {}
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        model.fit(train_features, parts.train.label)
        validation[name], _ = score_model(model, validation_features, parts.validation.label)
        print(f"  validation macro F1: {validation[name]['macro_f1']:.4f}", flush=True)
    selected = choose_model(validation)
    print(f"Selected by validation macro F1: {selected}", flush=True)
    return FittedExperiment(vectorizer, models, validation, selected)
