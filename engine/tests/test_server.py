# SPDX-License-Identifier: GPL-3.0-or-later
"""HTTP routes that changed: serving key frames and dense picture samples."""

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from vscrubber import library, server  # noqa: E402


@pytest.fixture
def client(data_dir, monkeypatch):
    lib = library.Library()
    lib.videos["v1"] = {"id": "v1", "path": "/nowhere.mp4", "name": "x.mp4", "status": "ready"}
    d = lib.dir("v1")
    (d / "frames").mkdir(parents=True)
    (d / "dense").mkdir()
    (d / "frames" / "0001.jpg").write_bytes(b"key")
    (d / "dense" / "d00012.jpg").write_bytes(b"dense")
    monkeypatch.setattr(server, "lib", lib)
    monkeypatch.setattr(server, "TOKEN", "")
    return TestClient(server.app)  # no `with`: skips startup work (hardware probing, model warm-up)


def test_serves_key_frames_and_dense_samples(client):
    r = client.get("/videos/v1/frames/0001.jpg")
    assert r.status_code == 200 and r.content == b"key" and r.headers["content-type"] == "image/jpeg"
    r = client.get("/videos/v1/frames/d00012.jpg")
    assert r.status_code == 200 and r.content == b"dense"


@pytest.mark.parametrize("name", ["..%2Fsecret.jpg", "x0001.jpg", "0001.png", "d1.jpg", "d000123.jpg"])
def test_rejects_other_names(client, name):
    assert client.get(f"/videos/v1/frames/{name}").status_code in (400, 404)


def test_missing_frame_and_video(client):
    assert client.get("/videos/v1/frames/0002.jpg").status_code == 404
    assert client.get("/videos/nope/frames/0001.jpg").status_code == 404


def test_token_required(client, monkeypatch):
    monkeypatch.setattr(server, "TOKEN", "secret")
    assert client.get("/videos/v1/frames/0001.jpg").status_code == 401
    assert client.get("/videos/v1/frames/0001.jpg", headers={"x-vs-token": "secret"}).status_code == 200
