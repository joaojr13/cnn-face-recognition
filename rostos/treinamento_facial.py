"""CNN facial congelada + camada neural supervisionada para os integrantes.

Não utiliza vizinho mais próximo, SVM ou comparação com uma galeria em inferência.
A identidade vem da saída linear + softmax treinada por entropia cruzada.
"""
from .arquivos import write_report
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .config import ARTIFACTS, REPORTS

import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1

FORMAT = "torch_facenet_v1"
PREPROCESSING = "face_pad160_rgb_fixed_standardization_v1"
NORMALIZATION = "(pixel − 127,5) / 128"
WEIGHTS_URL = "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
torch.set_num_threads(4)


def face_tensor(batch):
    pixels = np.asarray(batch, dtype=np.float32)
    if pixels.ndim != 4 or pixels.shape[1:] != (160, 160, 3):
        raise ValueError("A CNN facial espera imagens RGB de 160 × 160 pixels.")
    # Normalização oficial desta rede, diferente da usada pela MobileNetV2.
    return (torch.from_numpy(pixels.copy()).permute(0, 3, 1, 2) - 127.5) / 128.0


class FaceClassifier(nn.Module):
    def __init__(self, class_count, pretrained=False):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained="vggface2" if pretrained else None,
                                         classify=False)
        # Descarta a antiga saída das identidades VGGFace2 (não é usada).
        self.backbone.logits = nn.Identity()
        self.backbone.requires_grad_(False)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, class_count))
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        # Congela também as estatísticas do BatchNorm e o dropout da base.
        self.backbone.eval()
        return self

    def forward(self, normalized_images):
        return self.classifier(self.backbone(normalized_images))

    def predict_numpy(self, batch):
        self.eval()
        with torch.inference_mode():
            return self(face_tensor(batch)).softmax(dim=1).cpu().numpy()

    def inspect_numpy(self, batch):
        """Captura ativações reais na mesma passagem que calcula a previsão."""
        self.eval()
        captured = []
        handle = self.backbone.conv2d_1a.register_forward_hook(
            lambda module, inputs, output: captured.append(output[:, :8].detach().cpu().numpy()))
        try:
            with torch.inference_mode():
                features = self.backbone(face_tensor(batch))
                scores = self.classifier(features).softmax(dim=1).cpu().numpy()
            return scores, captured[0], int(features.shape[1])
        finally:
            handle.remove()


def load_model(path, class_count):
    # A aplicação carrega somente tensores; não desserializa objetos Python.
    model = FaceClassifier(class_count)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def training_variants(image):
    """Nove entradas derivadas de uma foto; não são nove observações independentes."""
    original = Image.fromarray(image)
    variants = [original, ImageOps.mirror(original), ImageOps.grayscale(original).convert("RGB")]
    variants += [original.rotate(angle, resample=Image.Resampling.BILINEAR)
                 for angle in (-4, 4)]
    variants += [ImageEnhance.Brightness(original).enhance(factor) for factor in (0.85, 1.15)]
    variants += [ImageEnhance.Contrast(original).enhance(factor) for factor in (0.85, 1.15)]
    return np.stack([np.asarray(item) for item in variants])


def train_specialized(epochs=200, threshold=0.9, activate=False):
    from .experimento_mobilenet import experimental_records
    from .treinamento_mobilenet import prepare_records
    from .modelo import decisions

    labels, records = experimental_records()
    counts = Counter(record["person"] for record in records)
    run_id = "facenet_experimental_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    run_dir, run_report = ARTIFACTS / run_id, REPORTS / run_id
    run_dir.mkdir()
    run_report.mkdir()
    images = prepare_records(records, run_report)
    write_report(run_report / "manifest.json", records)

    torch.manual_seed(42)
    np.random.seed(42)
    model = FaceClassifier(len(labels), pretrained=True)
    augmented = np.concatenate([training_variants(image) for image in images])
    targets = torch.tensor([labels.index(record["person"]) for record in records
                            for _ in range(9)], dtype=torch.long)
    print(f"CNN facial: {len(images)} fotos originais, {len(augmented)} entradas com variações; base congelada.", flush=True)
    # A base é fixa: calcular suas saídas uma vez equivale a reutilizá-las a cada
    # época. A inferência continua executando a CNN inteira sobre cada rosto.
    with torch.no_grad():
        features = torch.cat([model.backbone(face_tensor(augmented[i:i + 12]))
                              for i in range(0, len(augmented), 12)])
    class_weights = torch.tensor([len(records) / (len(labels) * counts[label]) for label in labels])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.03, weight_decay=0.001)
    model.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model.classifier(features), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(f"Época {epoch + 1}/{epochs} · perda de treino: {loss.item():.5f}", flush=True)
    model.eval()
    path = run_dir / "model.pt"
    torch.save(model.state_dict(), path)
    restored = load_model(path, len(labels))
    probabilities = restored.predict_numpy(images)
    np.testing.assert_allclose(probabilities, model.predict_numpy(images), atol=1e-6)
    predictions = decisions(probabilities, labels, threshold)
    diagnostics = [{"path": record["path"], "training_label": record["person"],
                    **prediction, "scores_by_class": dict(zip(labels, map(float, scores)))}
                   for record, prediction, scores in zip(records, predictions, probabilities)]
    write_report(run_report / "training_diagnostics.json", {
        "warning": "Diagnóstico nas fotos de treino; não é avaliação independente.",
        "predictions": diagnostics})
    write_report(run_report / "history.json", {"loss": losses})
    from pathlib import Path
    weights_path = Path(torch.hub.get_dir()).parent / "checkpoints/20180402-114759-vggface2.pt"
    metadata = {"run_id": run_id, "experimental": True, "model_format": FORMAT,
                "labels": labels, "threshold": threshold, "threshold_calibrated": False,
                "evaluation_available": False, "training_images": len(records),
                "class_counts": dict(counts), "image_size": 160,
                "architecture": "InceptionResnetV1 / VGGFace2 / base congelada",
                "preprocessing": PREPROCESSING, "normalization": NORMALIZATION,
                "detector": "MTCNN 1.0 / PNet: escalas com lado mínimo de 12 pixels",
                "classifier": "camada linear treinada + softmax",
                "augmentation": {"variants_per_image": 9, "independent_images": len(records),
                                 "operations": ["original", "espelhamento", "tons de cinza",
                                                "rotação ±4°", "brilho ±15%", "contraste ±15%"]},
                "epochs_completed": epochs, "seed": 42, "torch_version": str(torch.__version__),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "weights_source": WEIGHTS_URL,
                "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "limitations": "Experimento sem validação independente. Desconhecidos não calibrados."}
    write_report(run_dir / "metadata.json", metadata)
    if activate:
        activate_run(metadata)
    print(f"Execução salva: {run_id}; ativa: {activate}", flush=True)
    for result in diagnostics:
        print(f"Foto de treino {result['training_label']}: {result['label']} ({result['score']:.4f})")
    return metadata


def activate_run(metadata):
    temporary = ARTIFACTS / f"{metadata['run_id']}.tmp"
    write_report(temporary, metadata)
    temporary.replace(ARTIFACTS / "active.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--activate", action="store_true", help="Ativa somente após salvar e verificar o modelo.")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 1000 or not 0.5 < args.threshold <= 1:
        parser.error("Use 1–1000 épocas e um limiar maior que 0,5 e até 1.")
    train_specialized(args.epochs, args.threshold, args.activate)


if __name__ == "__main__":
    main()
