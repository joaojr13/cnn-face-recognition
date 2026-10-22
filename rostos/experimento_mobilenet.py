"""Experimento com toda a base disponível, sem validação ou teste independente.

Uso: .venv/bin/python -m rostos.experimento_mobilenet --epochs 80
Os resultados não substituem a avaliação acadêmica de rostos.treinamento_mobilenet.
"""
from .arquivos import write_report
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from .config import ARTIFACTS, DATA, REPORTS
from .dados import images_in, validate_duplicates
from .modelo import build_model, decisions
from .treinamento_mobilenet import prepare_records


def experimental_records(root=DATA / "raw"):
    records = []
    people = sorted(p for p in root.iterdir() if p.is_dir() and images_in(p))
    if len(people) < 2:
        raise ValueError("O experimento precisa de pelo menos duas pessoas com imagens.")
    labels = [p.name for p in people]
    if any(label.casefold() == "desconhecido" for label in labels):
        raise ValueError("Desconhecido é um nome reservado para rejeições.")
    for person in people:
        for path in images_in(person):
            records.append({"path": str(path), "person": person.name,
                            "session": str(path.parent.relative_to(person)),
                            "split": "experimental_train_only"})
    validate_duplicates(records)
    return labels, records


def train_experimental(epochs=80, threshold=0.9):
    labels, records = experimental_records()
    counts = Counter(record["person"] for record in records)
    run_id = "experimental_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    run_dir, run_report = ARTIFACTS / run_id, REPORTS / run_id
    run_dir.mkdir()
    run_report.mkdir()
    images = prepare_records(records, run_report)
    write_report(run_report / "manifest.json", records)

    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    model = build_model(len(labels))
    targets = np.array([labels.index(record["person"]) for record in records])
    # Compensa quantidades desiguais por pessoa sem contar cópias como novas observações.
    weights = np.array([len(records) / (len(labels) * counts[record["person"]])
                        for record in records], dtype="float32")
    ds = tf.data.Dataset.from_tensor_slices((images, targets, weights))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    ds = ds.shuffle(len(records), seed=42).batch(min(8, len(records))).with_options(options).prefetch(1)
    print(f"EXPERIMENTAL: {len(records)} imagens / {len(labels)} pessoas; sem validação ou teste.", flush=True)
    history = model.fit(ds, epochs=epochs, shuffle=False, verbose=2)
    model.save(run_dir / "model.keras")
    # Verificação nas próprias imagens de treino, registrada como diagnóstico.
    # Nunca gera evaluation.json, matriz de confusão ou métricas de generalização.
    predictions = decisions(model(images, training=False).numpy(), labels, threshold)
    diagnostics = [{"path": record["path"], "training_label": record["person"], **prediction}
                   for record, prediction in zip(records, predictions)]
    write_report(run_report / "training_diagnostics.json", {
        "warning": "Mesmas fotos usadas no treino; não é teste nem avaliação independente.",
        "predictions": diagnostics})
    write_report(run_report / "history.json", {key: [float(v) for v in values] for key, values in history.history.items()})
    metadata = {"run_id": run_id, "experimental": True, "labels": labels,
                "threshold": threshold, "threshold_calibrated": False,
                "evaluation_available": False, "training_images": len(records),
                "class_counts": dict(counts), "image_size": 160,
                "architecture": "MobileNetV2 / ImageNet / base congelada",
                "epochs_completed": len(history.epoch), "seed": 42,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tensorflow_version": tf.__version__,
                "limitations": "Treinado em toda a base disponível. Sem validação, teste independente ou calibração de desconhecidos."}
    write_report(run_dir / "metadata.json", metadata)
    temp = ARTIFACTS / f"{run_id}.tmp"
    write_report(temp, metadata)
    temp.replace(ARTIFACTS / "active.json")
    print(f"Experimento ativado: {run_id}. Limiar provisório: {threshold}.", flush=True)
    for row in diagnostics:
        print(f"Foto de treino ({row['training_label']}): {row['label']} / pontuação {row['score']:.4f}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Limiar provisório, não validado com pessoas externas.")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 200 or not 0.5 < args.threshold <= 1:
        parser.error("Use 1–200 épocas e limiar maior que 0,5 e até 1.")
    try:
        train_experimental(args.epochs, args.threshold)
    except ValueError as exc:
        parser.exit(2, f"Experimento não concluído: {exc}\n")


if __name__ == "__main__":
    main()
