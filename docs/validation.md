# Evaluation record

## Local execution

The 100,000-row Sentiment140 sample was downloaded and run locally on Python 3.12.14 with the versions pinned in `requirements.txt`. The selected Logistic Regression model achieved 79.37% accuracy and 0.7937 macro F1 on 9,720 held-out examples. On 359 binary examples from the official manual test file, it achieved 79.94% accuracy and 0.7986 macro F1.

The [recorded manifest](../results/benchmark/metrics.json) includes full-precision results, source checksums and split counts. The saved pipeline reproduced all held-out predictions after reloading. A separate prediction command returned positive for `I really enjoyed this film`, and negative for `The service was awful` and `I am not happy with this purchase`. Those three examples check inference, not general accuracy.

## Regression tests

All 19 tests passed locally. The suite checks negation, URL/username handling, normalization, duplicate and conflicting-label removal, disjoint stratified splits, vocabulary isolation from validation/test text, validation-based model selection, model serialization in a fresh process, import behavior and the CLI's metrics/predictions consistency. Download tests cover fallback, checksum rejection and extraction of only the expected CSV files.

## Continuous integration

GitHub Actions installs the pinned packages, runs the test suite, executes the offline synthetic demo and loads its saved pipeline in a second command. This keeps the default checks independent of dataset-host availability. The checked-in full-data-source sample benchmark was run locally; the synthetic CI run does not reproduce its accuracy.

The dev container and VS Code tasks provide the same commands. A hosted Codespaces session has not been manually tested.
