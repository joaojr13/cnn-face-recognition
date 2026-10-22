import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rostos.dados import dataset_records, unknown_records, validate_duplicates
from rostos.imagem import read_image, model_image, crop_face
from rostos.modelo import decisions
from rostos.treinamento_mobilenet import calibrate, report_metrics
from rostos import pipeline as server


def jpeg(size=(120, 100)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "#aabbcc").save(buffer, "JPEG")
    return buffer.getvalue()


def test_invalid_upload_and_exif_rotation():
    with pytest.raises(ValueError):
        read_image(b"not an image")
    with pytest.raises(ValueError):
        read_image(b"x" * (12 * 1024 * 1024 + 1))
    image = Image.new("RGB", (120, 80))
    exif = image.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    assert read_image(buffer.getvalue()).size == (80, 120)


def test_crops_clamp_to_bounds_and_preserve_proportions():
    image = Image.new("RGB", (100, 200), "white")
    crop = crop_face(image, [0, 0, 100, 200])
    assert crop.size == (100, 200)
    processed = np.array(model_image(crop))
    assert processed.shape == (160, 160, 3)
    assert processed[:, 0].max() == 0
    assert processed[:, 80].min() == 255


def test_phone_jpeg_with_auxiliary_frames_uses_primary_image():
    buffer = io.BytesIO()
    Image.new("RGB", (120, 80), "red").save(buffer, "MPO", save_all=True,
                                          append_images=[Image.new("RGB", (120, 80), "blue")])
    result = read_image(buffer.getvalue())
    assert result.size == (120, 80)
    red, green, blue = result.getpixel((60, 40))
    assert red > 240 and green < 10 and blue < 10


def test_split_keeps_sessions_separate_and_requires_enough_data(tmp_path):
    for person in ("a", "b"):
        for session, count in (("01", 10), ("02", 3), ("03", 3)):
            folder = tmp_path / person / session
            folder.mkdir(parents=True)
            for i in range(count):
                (folder / f"{i}.jpg").write_bytes(f"{person}-{session}-{i}".encode())
    labels, records = dataset_records(tmp_path, expected=2)
    assert labels == ["a", "b"]
    assert {r["session"] for r in records if r["split"] == "train"} == {"01"}
    assert {r["session"] for r in records if r["split"] == "validation"} == {"02"}
    assert {r["session"] for r in records if r["split"] == "test"} == {"03"}
    (tmp_path / "a" / "03" / "0.jpg").unlink()
    with pytest.raises(ValueError, match="mínimo"):
        dataset_records(tmp_path, expected=2)


def test_duplicate_across_splits_is_blocked(tmp_path):
    first, second = tmp_path / "one.jpg", tmp_path / "two.jpg"
    first.write_bytes(jpeg())
    second.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="duplicado"):
        validate_duplicates([{"path": str(first), "split": "train"}, {"path": str(second), "split": "test"}])


def test_external_identities_must_differ(tmp_path):
    for split in ("validation", "test"):
        folder = tmp_path / split / "same_person"
        folder.mkdir(parents=True)
        (folder / "photo.jpg").write_bytes(split.encode())
    with pytest.raises(ValueError, match="diferentes"):
        unknown_records(tmp_path)


def test_threshold_calibration_and_unknown_metrics():
    known = np.array([[0.95, 0.05], [0.1, 0.9]])
    external = np.array([[0.55, 0.45], [0.4, 0.6]])
    calibration = calibrate(known, [0, 1], external, ["a", "b"])
    threshold = calibration["threshold"]
    assert 0.6 < threshold <= 0.9
    metrics = report_metrics(np.concatenate([known, external]),
                             ["a", "b", "Desconhecido", "Desconhecido"], ["a", "b"], threshold)
    assert metrics["accuracy"] == 1
    assert metrics["unknown_false_accept_rate"] == 0
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    assert decisions([[0.8, 0.2]], ["a", "b"], 0.8)[0]["label"] == "a"


def test_untrained_pipeline_and_multiple_faces(monkeypatch):
    monkeypatch.setattr(server, "active_metadata", lambda: None)
    monkeypatch.setattr(server, "detect", lambda image: [
        {"box": [2, 2, 30, 40], "detection_score": .99},
        {"box": [60, 5, 90, 45], "detection_score": .98}])
    result = server.analyze(jpeg())
    assert len(result["faces"]) == 2
    assert all(f["score"] is None for f in result["faces"])
    with pytest.raises(ValueError):
        server.analyze(b'bad')
    monkeypatch.setattr(server, "detect", lambda image: [])
    assert server.analyze(jpeg())["faces"] == []


def test_experimental_never_exposes_independent_evaluation(tmp_path, monkeypatch):
    from rostos import estudo
    metadata = {"run_id": "demo", "experimental": True}
    (tmp_path / 'demo').mkdir()
    (tmp_path / 'demo/evaluation.json').write_text('{"accuracy":1}')
    monkeypatch.setattr(estudo, "REPORTS", tmp_path)
    assert not estudo.evaluation(metadata)['available']
    assert estudo.evaluation(None) == {'available': False}


def test_experiment_accepts_small_base_without_creating_fake_test_split(tmp_path):
    from rostos.experimento_mobilenet import experimental_records
    for person in ("a", "b"):
        folder = tmp_path / person / "sessao_01"
        folder.mkdir(parents=True)
        (folder / "photo.jpg").write_bytes(person.encode())
    labels, records = experimental_records(tmp_path)
    assert labels == ["a", "b"]
    assert len(records) == 2
    assert {record["split"] for record in records} == {"experimental_train_only"}
    # A flexibilização do experimento não muda as exigências do treino final.
    with pytest.raises(ValueError, match="sessões"):
        dataset_records(tmp_path, expected=2)


@pytest.mark.ml
def test_real_cnn_training_serialization_and_figures(tmp_path):
    import tensorflow as tf
    from rostos.modelo import build_model
    from rostos.treinamento_mobilenet import make_figures
    model = build_model(5)
    rng = np.random.default_rng(42)
    images = rng.integers(0, 256, (2, 160, 160, 3), dtype=np.uint8)
    before = model(images, training=False).numpy()
    assert before.shape == (2, 5)
    model.train_on_batch(images, np.array([0, 1]))
    after = model(images, training=False).numpy()
    assert not np.allclose(before, after)
    assert len(model.trainable_weights) == 2  # kernel + bias da saída
    path = tmp_path / "smoke.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(after, restored(images, training=False).numpy(), atol=1e-6)
    labels = ["a", "b", "c", "d", "e"]
    metrics = report_metrics(after, ["a", "Desconhecido"], labels, 0.9)
    make_figures({"loss": [1.0], "val_loss": [1.1], "accuracy": [0.5], "val_accuracy": [0.5]},
                 metrics, model, images[0], tmp_path)
    assert (tmp_path / "activation_maps.png").is_file()
    assert (tmp_path / "confusion_matrix.png").is_file()


@pytest.mark.ml
def test_complete_training_run_activates_only_after_evaluation(tmp_path, monkeypatch):
    """Exercita fit/validação/calibração/teste sem tocar na base do usuário."""
    from rostos import treinamento_mobilenet as training
    labels = ["a", "b", "c", "d", "e"]
    known, external = [], []
    for label in labels:
        for split in ("train", "validation", "test"):
            path = tmp_path / f"{label}_{split}.jpg"
            path.write_bytes(f"{label}_{split}".encode())
            known.append({"path": str(path), "person": label, "session": split, "split": split})
    for split in ("validation", "test"):
        path = tmp_path / f"external_{split}.jpg"
        path.write_bytes(split.encode())
        external.append({"path": str(path), "person": "Desconhecido", "session": split,
                         "external_identity": f"external_{split}", "split": split})
    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    monkeypatch.setattr(training, "ARTIFACTS", artifacts)
    monkeypatch.setattr(training, "REPORTS", reports)
    monkeypatch.setattr(training, "dataset_records", lambda: (labels, known))
    monkeypatch.setattr(training, "unknown_records", lambda: external)
    monkeypatch.setattr(training, "prepare_records", lambda records, folder:
                        np.random.default_rng(42).integers(0, 256, (len(records), 160, 160, 3), dtype=np.uint8))
    metadata = training.train(epochs=1, batch_size=5)
    import json
    saved = json.loads((artifacts / "active.json").read_text())
    assert saved["run_id"] == metadata["run_id"]
    metrics = json.loads((reports / saved["run_id"] / "evaluation.json").read_text())
    assert metrics["test_size"] == 6
    assert len(metrics["confusion_matrix"]) == 6
    assert (artifacts / saved["run_id"] / "model.keras").is_file()
