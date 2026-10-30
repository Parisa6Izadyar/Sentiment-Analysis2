# Data and methods

## Input and sampling

Read the headerless, Latin-1 Sentiment140 CSV with its six original fields. Use sentiment 0 as negative and 4 as positive. The training file contains 800,000 rows per class. Seed 42 selects 50,000 rows from each class before text cleaning. Metadata fields such as username, tweet ID and date are not model inputs.

The source archive SHA-256 is `004a3772c8a7ff9bbfeb875880f47f0679d93fc63e5cf9cff72d54a8a6162e57`. The source CSV checksums are recorded in the benchmark manifest. Source data are downloaded separately and not committed to this repository.

## Text processing and duplicates

The same tokenizer runs during duplicate checks, TF-IDF fitting and saved-model prediction. It decodes HTML, normalizes case, removes URLs and user mentions, expands common negative contractions, maps selected emoticons to words, reduces repeated characters and applies the NLTK Porter stemmer. Stopwords are retained, including negation. Porter stemming requires no external corpus files.

Of 100,000 sampled rows:

- 246 became empty after processing.
- 117 processed-text groups contained conflicting labels; all 729 rows in those groups were excluded.
- A further 1,830 rows duplicated another processed text with the same label.
- 97,195 distinct processed texts remained.

These checks use an explicit predeclared cleaning policy. Deduplication uses the exact token sequence seen by the model, not raw text alone. It does not remove paraphrases or establish semantic independence. Each partition also checks that it shares no processed text key with another partition.

## Model development

Split the cleaned sample into approximately 80% training, 10% validation and 10% test, stratifying by label with seed 42. Integer rounding gives 77,755 / 9,720 / 9,720 rows.

Fit word-unigram and word-bigram TF-IDF on training text only, using `min_df=2`, `max_features=100000` and float64 features. The run learned 88,475 features. Fit the three fixed classifiers on the same training matrix:

- Bernoulli Naive Bayes: alpha 2, with the estimator's default binarization at zero.
- Linear SVC: C 1, automatic primal/dual selection, maximum 5,000 iterations, seed 42.
- Logistic Regression: C 2, liblinear solver, maximum 1,000 iterations, seed 42.

Choose among the three using validation macro F1. Alphabetical model name resolves a tie. A most-frequent-class dummy model is reported as a baseline and excluded from selection. No test score is used to choose the saved model. The saved model remains fitted on the training partition; validation and test rows are not added to its training set.

After selection, measure each predefined candidate on the held-out test partition. Report accuracy, balanced accuracy, macro F1, ROC AUC, per-class precision/recall/F1 and a confusion matrix. Test scores describe this fixed experiment; repeatedly tuning against them would require a new independent test set.

## Official manual test file

The separate official file has 498 manually labelled examples: 359 positive/negative and 139 neutral. This project evaluates only the 359 binary examples. Their normalized text keys are checked before training; any development-row overlaps would be removed. None were found in this sample. The manual test labels are never used for fitting or model selection.

The selected pipeline scored 287/359 correct predictions (79.94%). The majority-class baseline scored 177/359 (49.30%). The manual set is small and from the same historical domain, so it does not establish performance on present-day social media.

## Prediction and reproducibility

Save the fitted vectorizer and selected classifier together as a scikit-learn Pipeline. Loading it back must reproduce every selected-model prediction on the internal test partition. An integration test also checks prediction in a fresh Python process.

The manifest records library versions, model parameters, row counts, cleaning counts, checksums and the seed. Split CSVs store original row positions and labels without publishing tweet text. Fixed data and package versions support reproduction; small numerical differences across platforms are possible.

The 120-example synthetic demo is only an offline integration fixture. Its metrics are not used in the README benchmark or CV.
