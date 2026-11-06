# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared fixtures. Tests need numpy, pillow and ffmpeg (or imageio-ffmpeg); no models or torch.

    cd engine && .venv/bin/python -m pytest tests     (after: uv pip install --python .venv/bin/python pytest httpx)
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vscrubber import catalog, media, settings  # noqa: E402

DEFAULTS = {"speech": "whisper-base", "caption": "smolvlm-256m", "vision": "clip-vit-b32", "text": "all-minilm-l6-v2"}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """An empty engine data folder, with fixed model choices so no hardware probing happens."""
    monkeypatch.setenv("VS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VS_SAVE_DIR", str(tmp_path / "save"))
    monkeypatch.setattr(catalog, "defaults", lambda: dict(DEFAULTS))
    monkeypatch.setattr(catalog, "_custom", None)
    monkeypatch.setattr(settings, "_cache", None)
    yield tmp_path / "data"
    settings._cache = None


def ffmpeg_or_skip() -> str:
    try:
        return media.ffmpeg()
    except RuntimeError:
        pytest.skip("ffmpeg isn't available")


@pytest.fixture(scope="session")
def clip(tmp_path_factory) -> Path:
    """12 s H.264/AAC MKV: red, blue, green for 4 s each (hard cuts at 4 s and 8 s), with a tone."""
    ff = ffmpeg_or_skip()
    out = tmp_path_factory.mktemp("media") / "cuts.mkv"
    colors = "".join(f"color=c={c}:s=320x180:r=30:d=4[{c}];" for c in ("red", "blue", "green"))
    r = subprocess.run([ff, "-v", "error", "-y", "-filter_complex", colors + "[red][blue][green]concat=n=3:v=1:a=0[v]",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=12", "-map", "[v]", "-map", "0:a",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("ffmpeg can't encode the test clip: " + r.stderr[-300:])
    return out
