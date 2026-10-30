import hashlib
from io import BytesIO
from zipfile import ZipFile

import pytest
from sentiment_analysis import download
from sentiment_analysis.data import TRAIN_FILE, TEST_FILE


def archive_bytes():
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as archive:
        archive.writestr(TRAIN_FILE, 'training fixture')
        archive.writestr(TEST_FILE, 'test fixture')
        archive.writestr('../outside.txt', 'must not be extracted')
    return buffer.getvalue()


def test_fallback_checks_archive_and_extracts_only_known_members(monkeypatch, tmp_path):
    content = archive_bytes()
    calls = []
    def fake_open(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise OSError('source unavailable')
        return BytesIO(content)
    monkeypatch.setattr(download, 'urlopen', fake_open)
    monkeypatch.setattr(download, 'ARCHIVE_SHA256', hashlib.sha256(content).hexdigest())
    train, test = download.download_data(tmp_path/'data')
    assert train.read_text() == 'training fixture'
    assert test.read_text() == 'test fixture'
    assert len(calls) == 2
    assert not (tmp_path/'outside.txt').exists()


def test_wrong_archive_is_rejected_before_extracting(tmp_path):
    (tmp_path/'trainingandtestdata.zip').write_bytes(archive_bytes())
    with pytest.raises(ValueError, match='checksum mismatch'):
        download.download_data(tmp_path)
    assert not (tmp_path/TRAIN_FILE).exists()
