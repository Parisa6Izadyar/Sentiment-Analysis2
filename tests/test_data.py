import pandas as pd
import pytest
from sentiment_analysis.data import clean_texts, split_texts, stratified_sample, read_sentiment140
from sentiment_analysis.text import text_key


def frame():
    return pd.DataFrame({'text': [f'example unique word{i}' for i in range(60)],
                         'label': [i % 2 for i in range(60)],
                         'source_row': range(60),
                         'text_key': [f'unique-{i}' for i in range(60)]})


def test_conflicts_and_normalized_duplicates_are_removed_together():
    raw = pd.DataFrame({'text': ['Love this', 'LOVE this!', 'bad day', 'BAD day',
                                'wonderful result', 'terrible result', None, '', '@user https://example.org'],
                        'label': [1, 1, 0, 1, 1, 0, 0, 0, 1], 'source_row': range(9)})
    cleaned, stats = clean_texts(raw)
    assert stats['duplicate_rows_removed'] == 1
    assert stats['conflicting_text_groups'] == 1
    assert stats['conflicting_rows_removed'] == 2
    assert stats['missing_or_blank_rows'] == 2
    assert stats['empty_after_processing'] == 1
    assert len(cleaned) == 3
    assert text_key('bad day') not in set(cleaned.text_key)


def test_split_preserves_rows_and_has_no_text_overlap():
    parts = split_texts(frame())
    all_parts = [parts.train, parts.validation, parts.test]
    assert [len(f) for f in all_parts] == [48, 6, 6]
    assert set(pd.concat(all_parts).source_row) == set(range(60))
    for i,a in enumerate(all_parts):
        assert set(a.label) == {0, 1}
        for b in all_parts[i+1:]:
            assert not set(a.text_key) & set(b.text_key)


def test_sampling_is_reproducible_and_stratified():
    a = stratified_sample(frame(), 30, 42)
    b = stratified_sample(frame(), 30, 42)
    assert list(a.source_row) == list(b.source_row)
    assert a.label.value_counts().to_dict() == {0: 15, 1: 15}


def test_duplicate_input_cannot_be_split():
    bad = frame();bad.loc[1, 'text_key'] = bad.loc[0, 'text_key']
    with pytest.raises(ValueError, match='Deduplicate'):
        split_texts(bad)


def test_unsupported_labels_are_rejected():
    bad = frame();bad.loc[0, 'label'] = 2
    with pytest.raises(ValueError, match='Labels'):
        clean_texts(bad)


def test_raw_loader_filters_neutral_only_for_manual_test(tmp_path):
    path = tmp_path/'raw.csv'
    path.write_text('0,1,date,query,user,bad\n4,2,date,query,user,good\n2,3,date,query,user,ordinary\n')
    with pytest.raises(ValueError, match='Expected Sentiment140'):
        read_sentiment140(path)
    loaded, info = read_sentiment140(path, allow_neutral=True)
    assert list(loaded.label) == [0, 1]
    assert info['file_rows'] == 3 and info['binary_rows'] == 2
