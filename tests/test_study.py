import hashlib
import io
import json

import pytest
from PIL import Image

from rostos import pipeline as server, estudo as study


def test_training_matrix_counts_confusion_and_rejection_without_fake_unknown_metrics():
    predictions = [{"training_label": a, "label": b} for a, b in
                   [("a", "a"), ("a", "b"), ("b", "Desconhecido"), ("b", "b")]]
    result = study.summarize_training(predictions, ["a", "b"])
    assert result["matrix"] == [[1, 1, 0], [0, 1, 1], [0, 0, 0]]
    assert result["accuracy"] == .5
    assert result["macro_precision"] == .75
    assert result["macro_recall"] == .5
    assert result["macro_f1"] == pytest.approx((2 / 3 + .5) / 2)
    assert result["rows"][-1]["f1"] is None
    assert result["unknown_false_accept_rate"] is None
    assert not result["independent"]


def test_gallery_uses_manifest_and_rejects_changed_or_external_files(tmp_path, monkeypatch):
    data, reports = tmp_path / "data", tmp_path / "reports"
    raw = data / "raw" / "a" / "session"
    raw.mkdir(parents=True)
    run = reports / "demo"
    run.mkdir(parents=True)
    stream = io.BytesIO()
    Image.new("RGB", (80, 100), "#aabbcc").save(stream, "JPEG")
    content = stream.getvalue()
    path = raw / "photo.jpg"
    path.write_bytes(content)
    record = {"path": str(path), "person": "a", "session": "session", "split": "experimental_train_only",
              "sha256": hashlib.sha256(content).hexdigest(), "box": [10, 10, 60, 80]}
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps([record]))
    metadata = {"run_id": "demo", "labels": ["a"], "experimental": True}
    monkeypatch.setattr(study, "DATA", data)
    monkeypatch.setattr(study, "REPORTS", reports)
    monkeypatch.setattr(server, "active_metadata", lambda: metadata)
    (raw / "added_later.jpg").write_bytes(content)
    summary = study.study_summary(metadata)
    assert len(summary['training_images']) == 1
    assert summary['training_diagnostic'] is None
    assert study.training_image(metadata, 0)['processed'].startswith('data:image/jpeg;base64,')
    with pytest.raises(IndexError):
        study.training_image(metadata, -1)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='mudou'):
        study.training_image(metadata, 0)
    external = tmp_path / 'outside.jpg'
    external.write_bytes(content)
    record['path'] = str(external)
    manifest.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match='não pertence'):
        study.training_image(metadata, 0)


@pytest.mark.ml
def test_inspection_uses_actual_convolution_and_preserves_predictions():
    import numpy as np
    import torch
    from rostos.treinamento_facial import FaceClassifier, face_tensor
    network = FaceClassifier(4).eval()
    pixels = np.random.default_rng(19).integers(0, 256, (2, 160, 160, 3), dtype=np.uint8)
    scores, maps, dimensions = network.inspect_numpy(pixels)
    np.testing.assert_allclose(scores, network.predict_numpy(pixels), atol=1e-6)
    with torch.inference_mode():
        expected = network.backbone.conv2d_1a(face_tensor(pixels))[:, :8].numpy()
    np.testing.assert_allclose(maps, expected, atol=1e-6)
    assert maps.shape == (2, 8, 79, 79)
    assert dimensions == 512
    assert not network.backbone.conv2d_1a._forward_hooks
