import hashlib
import io
import json

import pytest
from PIL import Image

from rostos import pipeline as server


def test_report_keeps_model_snapshot_and_explains_rejection(monkeypatch):
    metadata = {"run_id": "version_a", "labels": ["a", "b"], "threshold": .9, "experimental": True}
    calls = []
    def active():
        calls.append(1)
        return metadata if len(calls) == 1 else {**metadata, "run_id": "version_b", "threshold": .8}
    monkeypatch.setattr(server, "active_metadata", active)
    monkeypatch.setattr(server, "detect", lambda image: [{"box": [5, 5, 55, 65], "detection_score": .99}])
    def predict(batch, metadata):
        assert metadata["run_id"] == "version_a"
        return [{"label": "Desconhecido", "score": .895, "accepted": False,
                 "threshold": metadata["threshold"], "scores_by_class": {"a": .895, "b": .105}}]
    monkeypatch.setattr(server.recognizer, "predict", predict)
    buffer = io.BytesIO()
    Image.new("RGB", (80, 100), "red").save(buffer, "PNG")
    content = buffer.getvalue()
    data = server.analyze(content, "photo.png")
    report = data["report"]
    assert len(calls) == 1
    assert report["source"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert report["source"]["filename"] == "photo.png"
    assert report["model"]["run_id"] == data["model"]["run_id"] == "version_a"
    face = report["faces"][0]
    assert face["decision_reason"] == "below_threshold"
    assert face["margin_percentage_points"] == pytest.approx(-.5)
    assert face["label"] == "Desconhecido" and face["top_candidate"] == "a"
    assert "data:image" not in json.dumps(report)
    assert len(face["input_sha256"]) == 64


def test_report_with_no_faces(monkeypatch):
    monkeypatch.setattr(server, "active_metadata", lambda: None)
    monkeypatch.setattr(server, "detect", lambda image: [])
    buffer = io.BytesIO()
    Image.new("RGB", (80, 80), "white").save(buffer, "PNG")
    result = server.analyze(buffer.getvalue(), "photo.png")
    assert result["report"]["faces"] == []
    assert result["report"]["model"] is None
