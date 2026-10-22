"""Avaliação facial: protocolo testado com características sintéticas, sem a CNN real."""
import io
import json

import numpy as np
import pytest
from PIL import Image

from rostos import avaliacao_facial as evaluation
from rostos import estudo

LABELS = ["ana", "bia", "caio"]


def centroid_head(features, targets):
    """Camada falsa: softmax da semelhança com a média de cada classe."""
    features, targets = np.asarray(features), np.asarray(targets)
    return np.stack([features[targets == c].mean(axis=0) for c in range(len(LABELS))])


def centroid_predict(head, features):
    logits = 8 * np.asarray(features) @ head.T
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def photos(rng, targets, noise=0.05):
    base = np.eye(len(LABELS), 4)[targets]
    return base[:, None, :] + rng.normal(0, noise, (len(targets), 3, 4))


def test_leave_one_out_never_trains_with_the_predicted_photo():
    augmented = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    seen = []

    def fit(features, targets):
        seen.append({tuple(row) for row in features})
        assert len(features) == 6 and len(targets) == 6
        return None

    result = evaluation.leave_one_out_probabilities(
        augmented, np.array([0, 0, 1, 1]), fit, lambda head, x: np.ones((len(x), 2)) / 2)
    assert result.shape == (4, 2)
    for index, used in enumerate(seen):
        assert not used & {tuple(row) for row in augmented[index]}


def test_metrics_without_external_photos_do_not_invent_false_accepts_or_zero_f1():
    probabilities = np.array([[.95, .03, .02], [.4, .35, .25], [.02, .96, .02]])
    block = evaluation.metrics_block(probabilities, ["ana", "ana", "bia"], LABELS, .9, ["a", "b", "c"])
    assert block["confusion_matrix"][0] == [1, 0, 0, 1]
    assert block["unknown_false_accept_rate"] is None
    assert block["macro_classes"] == ["ana", "bia"]
    assert block["macro_f1"] == pytest.approx((2 / 3 + 1) / 2)
    assert block["samples"][1] == {"path": "b", "truth": "ana", "prediction": "Desconhecido",
                                   "top_candidate": "ana", "score": .4}
    json.dumps(block, allow_nan=False)


def test_protocol_calibrates_on_validation_and_reports_independent_test():
    rng = np.random.default_rng(3)
    train_targets = np.array([0, 0, 1, 1, 2, 2])
    test_targets = np.array([0, 1, 2])
    train = {"augmented": photos(rng, train_targets), "targets": train_targets,
             "paths": [f"train/{i}" for i in range(6)]}
    test_aug = photos(rng, test_targets)
    active = centroid_head(train["augmented"].reshape(-1, 4), np.repeat(train_targets, 3))
    member_test = {"augmented": test_aug, "targets": test_targets, "paths": ["t/a", "t/b", "t/c"],
                   "probabilities": centroid_predict(active, test_aug[:, 0])}

    def external(count):
        return {"probabilities": centroid_predict(active, rng.normal(0, .05, (count, 4))),
                "paths": [f"lfw/{i}" for i in range(count)]}

    result = evaluation.run_protocol(LABELS, train, member_test, external(20), external(20),
                                     centroid_head, centroid_predict, provisional_threshold=.9)
    assert result["labels"] == LABELS + ["Desconhecido"]
    assert result["calibration"]["validation_unknown_false_accept_rate"] <= .05
    assert result["calibration"]["known_validation_images"] == 6
    test = result["test"]
    assert test["known_images"] == 3 and test["unknown_images"] == 20
    assert test["threshold"] == result["threshold"]
    assert result["test_at_provisional_threshold"]["threshold"] == .9
    assert test["accuracy"] == 1.0 and test["unknown_false_accept_rate"] == 0.0
    assert result["cross_validation"]["known_images"] == 9
    assert result["cross_validation"]["unknown_false_accept_rate"] is None
    json.dumps(result, allow_nan=False)


def test_threshold_simulation_reproduces_calibration_and_explains_the_trade_off():
    rng = np.random.default_rng(5)
    train_targets = np.array([0, 0, 1, 1, 2, 2])
    train = {"augmented": photos(rng, train_targets, .35), "targets": train_targets,
             "paths": [f"train/{i}" for i in range(6)]}
    test_aug = photos(rng, np.array([0, 1, 2]), .35)
    active = centroid_head(train["augmented"].reshape(-1, 4), np.repeat(train_targets, 3))
    member_test = {"augmented": test_aug, "targets": np.array([0, 1, 2]), "paths": ["a", "b", "c"],
                   "probabilities": centroid_predict(active, test_aug[:, 0])}

    def external(count):
        return {"probabilities": centroid_predict(active, rng.normal(.3, .4, (count, 4))),
                "paths": [f"lfw/{i}" for i in range(count)]}

    result = json.loads(json.dumps(evaluation.run_protocol(
        LABELS, train, member_test, external(30), external(30), centroid_head, centroid_predict, .9)))
    samples = result["validation"]["samples"]
    chosen = evaluation.threshold_effect(samples, LABELS, result["threshold"])
    assert chosen["macro_f1"] == pytest.approx(result["calibration"]["validation_macro_f1"])
    assert chosen["false_accept_rate"] == pytest.approx(
        result["calibration"]["validation_unknown_false_accept_rate"])
    test = evaluation.threshold_effect(result["test"]["samples"], LABELS, result["threshold"])
    assert test["members_correct"] + test["members_rejected"] + test["members_wrong_name"] == 3
    assert test["outsiders_accepted"] == round(result["test"]["unknown_false_accept_rate"] * 30)
    low = evaluation.threshold_effect(samples, LABELS, 0.0)
    high = evaluation.threshold_effect(samples, LABELS, 1.01)
    assert low["members_rejected"] == 0 and high["outsiders_accepted"] == 0
    assert high["members_rejected"] == high["members"]
    candidates = evaluation.threshold_candidates(result)
    values = [value for value, _ in candidates]
    assert values == sorted(values)
    assert any("calibrado" in note for _, note in candidates)
    assert any("provisório" in note for _, note in candidates)


def test_threshold_curve_figure_renders(tmp_path):
    from rostos.graficos import save, threshold_curve_figure
    samples = [{"truth": "ana", "top_candidate": "ana", "score": .97},
               {"truth": "Desconhecido", "top_candidate": "bia", "score": .93}]
    rows = [evaluation.threshold_effect(samples, LABELS, t) for t in np.linspace(.7, 1, 31)]
    save(threshold_curve_figure(rows, .95, .9, .05), tmp_path / "curva.png")
    assert Image.open(tmp_path / "curva.png").size[0] > 800


def jpeg(color):
    stream = io.BytesIO()
    Image.new("RGB", (40, 40), color).save(stream, "JPEG")
    return stream.getvalue()


def test_member_test_records_require_every_member_and_reject_unknown_folders(tmp_path):
    for index, label in enumerate(LABELS):
        (tmp_path / label).mkdir()
        (tmp_path / label / "nova.jpg").write_bytes(jpeg((index * 60, 20, 20)))
    records = evaluation.member_test_records(LABELS, tmp_path)
    assert [r["person"] for r in records] == LABELS and all(r["sha256"] for r in records)
    (tmp_path / "intruso").mkdir()
    with pytest.raises(ValueError, match="intruso"):
        evaluation.member_test_records(LABELS, tmp_path)
    (tmp_path / "intruso").rmdir()
    (tmp_path / "caio" / "nova.jpg").unlink()
    with pytest.raises(ValueError, match="caio"):
        evaluation.member_test_records(LABELS, tmp_path)


def test_test_photo_identical_to_training_photo_is_rejected():
    with pytest.raises(ValueError, match="idêntica"):
        evaluation.check_no_overlap([{"sha256": "x"}], [{"sha256": "x"}])


def test_apply_threshold_keeps_weights_and_provisional_history(tmp_path):
    run = tmp_path / "demo"
    run.mkdir()
    metadata = {"run_id": "demo", "threshold": .9, "threshold_calibrated": False, "labels": LABELS}
    (run / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "active.json").write_text(json.dumps(metadata))
    result = {"threshold": .61, "test": {"known_images": 5, "unknown_images": 50},
              "calibration": {"threshold": .61, "target_max_far": .05, "validation_macro_f1": .8,
                              "validation_unknown_false_accept_rate": .04, "known_validation_images": 10,
                              "unknown_validation_images": 50, "method": "m", "note": "n"}}
    updated = evaluation.apply_threshold("demo", result, tmp_path)
    active = json.loads((tmp_path / "active.json").read_text())
    assert active == updated
    assert active["threshold"] == .61 and active["provisional_threshold"] == .9
    assert active["threshold_calibrated"] and active["evaluation_available"]
    result["threshold"] = .7
    assert evaluation.apply_threshold("demo", result, tmp_path)["provisional_threshold"] == .9


def test_study_reads_evaluation_of_experimental_model(tmp_path, monkeypatch):
    monkeypatch.setattr(estudo, "REPORTS", tmp_path)
    metadata = {"run_id": "demo", "experimental": True}
    assert estudo.evaluation(metadata)["available"] is False
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "evaluation.json").write_text(json.dumps({"run_id": "demo", "test": {}}))
    assert estudo.evaluation(metadata)["available"] is True
    (tmp_path / "demo" / "evaluation.json").write_text(json.dumps({"run_id": "outro", "test": {}}))
    assert estudo.evaluation(metadata)["available"] is False


def test_figures_are_generated_from_saved_results(tmp_path):
    result = {"labels": LABELS + ["Desconhecido"], "threshold": .7, "provisional_threshold": .9}
    matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    sample = lambda truth, score: {"truth": truth, "score": score}
    for key in ("test", "cross_validation", "validation"):
        result[key] = {"confusion_matrix": matrix,
                       "samples": [sample("ana", .95), sample("Desconhecido", .5)]}
    names = evaluation.save_figures(result, tmp_path)
    for filename in names.values():
        assert Image.open(tmp_path / filename).size[0] > 500


@pytest.mark.ml
def test_head_training_matches_procedure_and_class_weights():
    import torch
    from rostos.treinamento_facial import fit_head, head_class_weights, head_probabilities
    weights = head_class_weights(torch.tensor([0, 0, 0, 1]), 2)
    assert weights.tolist() == pytest.approx([4 / 6, 2.0])
    with pytest.raises(ValueError):
        head_class_weights(torch.tensor([0, 0]), 2)
    rng = np.random.default_rng(1)
    features = np.concatenate([rng.normal(1, .1, (9, 512)), rng.normal(-1, .1, (9, 512))])
    targets = np.array([0] * 9 + [1] * 9)
    first = head_probabilities(fit_head(features, targets, 2, epochs=30), features)
    second = head_probabilities(fit_head(features, targets, 2, epochs=30), features)
    np.testing.assert_allclose(first, second)
    assert (first.argmax(axis=1) == targets).all()
