import json
from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from sentiment_analysis.data import ROOT, clean_texts, split_texts
from sentiment_analysis.experiment import choose_model, fit_candidates
from sentiment_analysis.text import tokenize_text


@pytest.fixture(scope='module')
def fitted():
    raw = pd.read_csv(ROOT/'data/demo.csv');raw['source_row'] = raw.index
    clean, _ = clean_texts(raw)
    parts = split_texts(clean)
    parts.validation['text'] += ' validationexclusive'
    parts.test['text'] += ' holdoutexclusive'
    experiment = fit_candidates(parts, min_df=1)
    return experiment, parts


def test_validation_and_test_words_cannot_enter_vocabulary(fitted):
    exp, _ = fitted
    vocab = exp.vectorizer.vocabulary_
    assert tokenize_text('validationexclusive')[0] not in vocab
    assert tokenize_text('holdoutexclusive')[0] not in vocab


def test_selection_uses_validation_f1_and_excludes_dummy():
    values = {'bernoulli_nb': {'macro_f1': 0.7}, 'linear_svc': {'macro_f1': 0.8},
              'logistic_regression': {'macro_f1': 0.75}, 'majority_baseline': {'macro_f1': 1.0}}
    assert choose_model(values) == 'linear_svc'


def test_predictions_survive_saving_and_fresh_process(fitted, tmp_path):
    exp, parts = fitted
    pipeline = Pipeline([('tfidf', exp.vectorizer), ('classifier', exp.models[exp.selected])])
    path = tmp_path/'model.joblib';joblib.dump(pipeline, path)
    texts = parts.test.text.tolist()
    expected = pipeline.predict(texts).tolist()
    result = subprocess.run([sys.executable, '-m', 'sentiment_analysis', 'predict',
                             '--model', str(path), '--text', *texts], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    actual = [1 if item['sentiment'] == 'positive' else 0 for item in json.loads(result.stdout)]
    assert actual == expected


def test_cli_demo_writes_consistent_metrics_and_predictions(tmp_path):
    result = subprocess.run([sys.executable, 'HS_Project_Z.py', 'demo', '--output', str(tmp_path)],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    predictions = pd.read_csv(tmp_path/'test_predictions.csv')
    assert metrics['run_type'] == 'synthetic integration demo'
    assert metrics['saved_pipeline_roundtrip_matches'] is True
    assert metrics['processed_text_overlap'] == 0
    for name, scores in metrics['test'].items():
        measured = (predictions[name] == predictions.label).mean()
        assert scores['accuracy'] == pytest.approx(measured)
        assert sum(sum(row) for row in scores['confusion_matrix']) == len(predictions)
    assert (tmp_path/'confusion_matrices.png').stat().st_size > 0
    assert 'Finished.' in result.stdout


def test_import_does_not_start_training_or_download_data():
    result = subprocess.run([sys.executable, '-c', 'import HS_Project_Z'], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    assert result.stdout == '' and result.stderr == ''
