"""Train, evaluate and use the sentiment classifiers."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .data import ROOT, TRAIN_FILE, TEST_FILE, checksum, clean_texts, read_sentiment140, stratified_sample, split_texts
from .download import download_data
from .evaluate import plot_confusions, score_model
from .experiment import fit_candidates


def run_training(frame, output, source, manual=None, manual_source=None, seed=42, max_features=100000, demo=False):
    start = time.monotonic()
    print(f"Cleaning {len(frame):,} sampled texts...", flush=True)
    frame, cleaning = clean_texts(frame)
    if manual is not None:
        manual, manual_cleaning = clean_texts(manual)
        overlap = frame.text_key.isin(set(manual.text_key))
        cleaning["manual_test_overlaps_removed"] = int(overlap.sum())
        frame = frame.loc[~overlap].copy()
        manual_source = {**manual_source, "cleaning": manual_cleaning}
    parts = split_texts(frame, seed)
    print(f"Rows: train {len(parts.train):,}; validation {len(parts.validation):,}; test {len(parts.test):,}", flush=True)
    fitted = fit_candidates(parts, seed, max_features=max_features, min_df=1 if demo else 2)
    features = fitted.vectorizer.transform(parts.test.text)
    heldout = {}
    predictions = parts.test[["source_row", "label"]].copy()
    for name, model in fitted.models.items():
        heldout[name], predictions[name] = score_model(model, features, parts.test.label)
        print(f"Test {name}: accuracy {heldout[name]['accuracy']:.4f}, macro F1 {heldout[name]['macro_f1']:.4f}", flush=True)
    report = {
        "run_type": "synthetic integration demo" if demo else "Sentiment140 sampled benchmark",
        "seed": seed, "source": source, "cleaning": cleaning,
        "split": {name: {"rows": len(getattr(parts, name)),
                         "class_counts": {str(k): int(v) for k, v in getattr(parts, name).label.value_counts().sort_index().items()}}
                  for name in ["train", "validation", "test"]},
        "processed_text_overlap": 0,
        "model_selection": "highest validation macro F1 among the three fixed classifiers; alphabetical tie break",
        "selected_model": fitted.selected,
        "features": len(fitted.vectorizer.vocabulary_),
        "feature_settings": {"ngram_range": [1, 2], "max_features": max_features, "min_df": 1 if demo else 2},
        "classifier_settings": {name: model.get_params() for name, model in fitted.models.items()},
        "validation": fitted.validation, "test": heldout,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "nltk", "joblib", "matplotlib"]},
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if manual is not None:
        manual_features = fitted.vectorizer.transform(manual.text)
        manual_metrics = {}
        for name in [fitted.selected, "majority_baseline"]:
            manual_metrics[name], _ = score_model(fitted.models[name], manual_features, manual.label)
        report["manual_test"] = {"source": manual_source, "rows": len(manual), "metrics": manual_metrics}
        print(f"Manual test: {len(manual)} binary examples; selected model accuracy {manual_metrics[fitted.selected]['accuracy']:.4f}", flush=True)
    pipeline = Pipeline([("tfidf", fitted.vectorizer), ("classifier", fitted.models[fitted.selected])])
    joblib.dump(pipeline, output / "sentiment_pipeline.joblib")
    restored = joblib.load(output / "sentiment_pipeline.joblib")
    if not (restored.predict(parts.test.text) == predictions[fitted.selected]).all():
        raise RuntimeError("Saved pipeline predictions differ from the evaluated model")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    for name in ["train", "validation", "test"]:
        # The split record identifies source rows without publishing tweet text.
        getattr(parts, name)[["source_row", "label"]].to_csv(output / f"{name}_rows.csv", index=False)
    plot_confusions(heldout, output / "confusion_matrices.png")
    report["saved_pipeline_roundtrip_matches"] = True
    report["elapsed_seconds"] = round(time.monotonic() - start, 3)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Finished. Results: {output.resolve()}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run a Sentiment140 benchmark")
    train.add_argument("--data", type=Path, default=ROOT / "data" / TRAIN_FILE)
    train.add_argument("--manual-test", type=Path)
    train.add_argument("--download", action="store_true", help="Download the original archive if needed")
    train.add_argument("--sample-size", type=int, default=100000, help="Stratified source rows; zero uses all rows")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-features", type=int, default=100000)
    train.add_argument("--output", type=Path, default=ROOT / "results/reproduced")
    demo = commands.add_parser("demo", help="Offline integration check using synthetic examples")
    demo.add_argument("--output", type=Path, default=ROOT / "results/demo")
    predict = commands.add_parser("predict", help="Predict with a saved local pipeline")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--text", nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            pipeline = joblib.load(args.model)
            predictions = pipeline.predict(args.text)
            print(json.dumps([{"text": text, "sentiment": "positive" if int(label) else "negative"}
                              for text, label in zip(args.text, predictions)], ensure_ascii=False, indent=2))
            return
        if args.command == "demo":
            frame = pd.read_csv(ROOT / "data/demo.csv")
            frame["source_row"] = frame.index
            print("Synthetic demo: checks execution only; scores are not a benchmark.", flush=True)
            run_training(frame, args.output, {"rows": len(frame), "sha256": checksum(ROOT / "data/demo.csv")},
                         max_features=2000, demo=True)
            return
        if args.max_features < 1:
            raise ValueError("max-features must be positive")
        if args.download:
            train_path, manual_path = download_data(ROOT / "data")
            args.data = train_path
            if args.manual_test is None:
                args.manual_test = manual_path
        if not args.data.is_file():
            raise ValueError(f"Dataset not found: {args.data}. Use train --download or provide --data.")
        print(f"Reading {args.data.name}...", flush=True)
        frame, source = read_sentiment140(args.data)
        frame = stratified_sample(frame, args.sample_size, args.seed)
        source["sampled_rows"] = len(frame)
        manual, manual_source = None, None
        if args.manual_test is not None:
            manual, manual_source = read_sentiment140(args.manual_test, allow_neutral=True)
        run_training(frame, args.output, source, manual, manual_source, seed=args.seed, max_features=args.max_features)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
