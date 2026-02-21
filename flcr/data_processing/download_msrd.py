from __future__ import annotations

import sys
import time
from urllib.request import urlretrieve

from flcr.config import RAW_MSRD_MOVIES_PATH, RAW_MSRD_QUERIES_PATH, ensure_directories


MSRD_MOVIES_URL = "https://media.githubusercontent.com/media/metarank/msrd/master/dataset/movies.csv.gz"
MSRD_QUERIES_URL = "https://media.githubusercontent.com/media/metarank/msrd/master/dataset/queries.csv.gz"


def _format_size(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def make_progress_hook(label: str):
    start = time.time()
    last_percent = -1

    def reporthook(block_count: int, block_size: int, total_size: int):
        nonlocal last_percent
        downloaded = block_count * block_size
        if total_size > 0:
            downloaded = min(downloaded, total_size)
        elapsed = max(time.time() - start, 1e-6)
        speed = downloaded / elapsed
        if total_size > 0:
            percent = int(downloaded * 100 / total_size)
            if percent == last_percent and downloaded < total_size:
                return
            last_percent = percent
            msg = (
                f"\r{label}: {percent:3d}% "
                f"({_format_size(downloaded)} / {_format_size(total_size)}) "
                f"at {_format_size(speed)}/s"
            )
        else:
            msg = f"\r{label}: {_format_size(downloaded)} at {_format_size(speed)}/s"
        sys.stdout.write(msg)
        sys.stdout.flush()
        if total_size > 0 and downloaded >= total_size:
            sys.stdout.write("\n")

    return reporthook


def maybe_download(url: str, path) -> None:
    if path.exists():
        print(f"Using existing raw file: {path}")
        return
    print(f"Downloading {url}")
    urlretrieve(url, path, reporthook=make_progress_hook(path.name))
    print(f"Saved to {path}")


def main():
    ensure_directories()
    maybe_download(MSRD_MOVIES_URL, RAW_MSRD_MOVIES_PATH)
    maybe_download(MSRD_QUERIES_URL, RAW_MSRD_QUERIES_PATH)


if __name__ == "__main__":
    main()
