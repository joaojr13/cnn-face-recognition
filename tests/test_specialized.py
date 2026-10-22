import numpy as np
import pytest

from rostos import modelo as model


def test_model_formats_and_artifact_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "ARTIFACTS", tmp_path)
    assert model.artifact_path({"run_id": "old"}) == tmp_path / "old/model.keras"
    assert model.artifact_path({"run_id": "new", "model_format": "torch_facenet_v1"}) == tmp_path / "new/model.pt"
    for metadata in ({"run_id": "../escape"}, {"run_id": "x", "model_format": "unsupported"}):
        with pytest.raises(ValueError):
            model.artifact_path(metadata)


@pytest.mark.ml
def test_facial_cnn_freezing_normalization_and_roundtrip(tmp_path):
    import torch
    from rostos.treinamento_facial import FaceClassifier, face_tensor, load_model

    pixels = np.random.default_rng(8).integers(0, 256, (2, 160, 160, 3), dtype=np.uint8)
    normalized = face_tensor(pixels)
    assert normalized.shape == (2, 3, 160, 160)
    assert face_tensor(np.zeros_like(pixels)).min().item() == -127.5 / 128
    assert face_tensor(np.full_like(pixels, 255)).max().item() == 127.5 / 128
    with pytest.raises(ValueError):
        face_tensor(pixels[:, :80])
    network = FaceClassifier(4).train()
    assert not network.backbone.training
    assert all(not p.requires_grad for p in network.backbone.parameters())
    backbone_before = {name: tensor.clone() for name, tensor in network.backbone.state_dict().items()}
    head_before = network.classifier[1].weight.detach().clone()
    optimizer = torch.optim.Adam(network.classifier.parameters(), lr=0.01)
    optimizer.zero_grad()
    torch.nn.functional.cross_entropy(network(normalized), torch.tensor([0, 1])).backward()
    optimizer.step()
    assert not torch.equal(head_before, network.classifier[1].weight)
    assert all(torch.equal(tensor, backbone_before[name]) for name, tensor in network.backbone.state_dict().items())
    before = network.predict_numpy(pixels)
    path = tmp_path / "model.pt"
    torch.save(network.state_dict(), path)
    after = load_model(path, 4).predict_numpy(pixels)
    np.testing.assert_allclose(after, before, atol=1e-6)
    np.testing.assert_allclose(after.sum(axis=1), 1, atol=1e-6)
