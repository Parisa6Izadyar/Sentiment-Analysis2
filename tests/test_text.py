import pytest
from sentiment_analysis.text import tokenize_text, text_key


@pytest.mark.parametrize('text', ["This is not good", "This isn't good", "This isn’t good"])
def test_negation_survives_processing(text):
    assert 'not' in tokenize_text(text)
    assert text_key(text) != text_key('This is good')


def test_urls_and_usernames_do_not_become_sentiment_features():
    assert text_key('@someperson I LOVE it https://example.org/awful') == text_key('I love it')


def test_html_case_and_emoticon_handling():
    assert text_key('GREAT &amp; lovely :-)') == text_key('great lovely smile')
    assert 'smile' in tokenize_text('Great :-D')


def test_nonstrings_are_rejected():
    with pytest.raises(TypeError):
        tokenize_text(None)
