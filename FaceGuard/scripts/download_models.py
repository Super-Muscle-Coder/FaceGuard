"""Download required FaceGuard ONNX models into scripts/models."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from config.settings import MODEL_URLS


MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "scrfd_10g_bnkps.onnx": MODEL_URLS["SCRFD"],
    "glintr100.onnx": MODEL_URLS["GLINTR100"],
}


def _download(url: str, target: Path):
    with urllib.request.urlopen(url, timeout=60) as response, open(target, "wb") as f:
        f.write(response.read())


def main() -> int:
    print(f"Models dir: {MODELS_DIR}")

    for filename, urls in TARGETS.items():
        target = MODELS_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"[SKIP] {filename} already exists")
            continue

        ok = False
        for url in urls:
            try:
                print(f"[DOWNLOAD] {filename} <- {url}")
                _download(url, target)
                print(f"[SUCCESS] {filename} ({target.stat().st_size / 1024 / 1024:.2f} MB)")
                ok = True
                break
            except Exception as ex:
                print(f"[WARNING] failed source: {url} ({ex})")

        if not ok:
            print(f"[ERROR] cannot download {filename}")
            return 1

    print("[DONE] All models ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
