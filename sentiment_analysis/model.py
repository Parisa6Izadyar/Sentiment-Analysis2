"""TF-IDF features and the three classifiers compared in the project."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

from .text import tokenize_text


def make_vectorizer(max_features=100000, min_df=2):
    return TfidfVectorizer(tokenizer=tokenize_text, token_pattern=None, lowercase=False,
                           ngram_range=(1, 2), max_features=max_features,
                           min_df=min_df, dtype=np.float64)


def make_classifiers(seed=42):
    return {
        "bernoulli_nb": BernoulliNB(alpha=2),
        "linear_svc": LinearSVC(C=1, dual="auto", max_iter=5000, random_state=seed),
        "logistic_regression": LogisticRegression(C=2, solver="liblinear", max_iter=1000,
                                                  random_state=seed),
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
    }
