"""Download the published Sentiment140 archive into the local data directory."""
from pathlib import Path
import shutil
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

from .data import TRAIN_FILE, TEST_FILE, checksum

ARCHIVE_SHA256 = "004a3772c8a7ff9bbfeb875880f47f0679d93fc63e5cf9cff72d54a8a6162e57"

URLS = [
    "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip",
    "https://mmlspark.blob.core.windows.net/publicwasb/twittersentimenttrainingandtestdata.zip",
]


def download_data(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if all((directory / name).is_file() for name in [TRAIN_FILE, TEST_FILE]):
        return directory / TRAIN_FILE, directory / TEST_FILE
    archive = directory / "trainingandtestdata.zip"
    if not archive.is_file():
        errors = []
        for url in URLS:
            partial = archive.with_suffix(".part")
            try:
                print(f"Downloading Sentiment140 from {url}", flush=True)
                with urlopen(url, timeout=30) as response, partial.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
                if checksum(partial) != ARCHIVE_SHA256:
                    raise ValueError("Downloaded archive does not match the recorded Sentiment140 checksum")
                with ZipFile(partial) as z:
                    for name in [TRAIN_FILE, TEST_FILE]:
                        z.getinfo(name)
                partial.replace(archive)
                break
            except (OSError, ValueError, BadZipFile) as error:
                partial.unlink(missing_ok=True)
                errors.append(f"{url}: {error}")
        else:
            raise RuntimeError("Dataset download failed. " + "; ".join(errors))
    if checksum(archive) != ARCHIVE_SHA256:
        raise ValueError(f"Archive checksum mismatch: {archive}. Remove it and download again.")
    print(f"Archive SHA-256: {ARCHIVE_SHA256}", flush=True)
    with ZipFile(archive) as z:
        # Extract only the two expected files, without trusting archive paths.
        for name in [TRAIN_FILE, TEST_FILE]:
            target = directory / name
            partial = target.with_suffix(".part")
            with z.open(name) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
            partial.replace(target)
    return directory / TRAIN_FILE, directory / TEST_FILE
