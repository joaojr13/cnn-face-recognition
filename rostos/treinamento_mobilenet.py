"""Treino supervisionado, calibração na validação e teste final independente."""
from .arquivos import write_report
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import ARTIFACTS, REPORTS
from .dados import dataset_records, unknown_records, validate_duplicates
from .imagem import crop_face, detect, model_image, read_image
from .modelo import build_model, decisions


def calibrate(known_probabilities, known_targets, unknown_probabilities, labels, max_far=0.05):
    """Escolhe o limiar só na validação, limitando aceites de pessoas externas."""
    truth = [labels[int(i)] for i in known_targets] + ["Desconhecido"] * len(unknown_probabilities)
    combined = np.concatenate([known_probabilities, unknown_probabilities])
    # Inclui um limiar acima de 1: rejeitar todos é preferível a inventar segurança.
    candidates = sorted(set([0.0, 1.000001] + [float(s) + 1e-7 for s in combined.max(axis=1)]))
    best = None
    for threshold in candidates:
        predicted = [d["label"] for d in decisions(combined, labels, threshold)]
        far = float(np.mean(unknown_probabilities.max(axis=1) >= threshold))
        macro = float(f1_score(truth, predicted, labels=labels + ["Desconhecido"],
                               average="macro", zero_division=0))
        if far <= max_far:
            candidate = (macro, -threshold, threshold, far)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return {"threshold": best[2], "validation_macro_f1": best[0],
            "validation_unknown_false_accept_rate": best[3], "target_max_far": max_far,
            "note": "Limiar empírico de validação; não garante rejeição de toda pessoa externa."}


def report_metrics(probabilities, truth, labels, threshold):
    predicted = [d["label"] for d in decisions(probabilities, labels, threshold)]
    all_labels = labels + ["Desconhecido"]
    report = classification_report(truth, predicted, labels=all_labels, output_dict=True, zero_division=0)
    known = np.array([name != "Desconhecido" for name in truth])
    accepted = np.array([name != "Desconhecido" for name in predicted])
    closed_predictions = [labels[int(i)] for i in np.asarray(probabilities)[known].argmax(axis=1)]
    return {"labels": all_labels,
            "confusion_matrix": confusion_matrix(truth, predicted, labels=all_labels).tolist(),
            "classification_report": report,
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "known_rejection_rate": float(np.mean(~accepted[known])),
            "unknown_false_accept_rate": float(np.mean(accepted[~known])),
            "known_closed_set_accuracy": float(accuracy_score(np.array(truth)[known], closed_predictions)),
            "test_size": len(truth), "predictions": predicted, "truth": truth}


def prepare_records(records, run_report: Path):
    arrays, failures = [], []
    for index, record in enumerate(records):
        print(f"Tratando imagem {index + 1}/{len(records)}: {Path(record['path']).name}", flush=True)
        try:
            image = read_image(Path(record["path"]).read_bytes())
            faces = detect(image)
            if len(faces) != 1:
                raise ValueError(f"Esperado exatamente 1 rosto; encontrados {len(faces)}.")
            crop = crop_face(image, faces[0]["box"])
            processed = model_image(crop)
            arrays.append(np.asarray(processed))
            record["box"] = faces[0]["box"]
        except ValueError as exc:
            failures.append({"path": record["path"], "reason": str(exc)})
    write_report(run_report / "preprocessing.json", {
        "total": len(records), "usable": len(arrays), "failures": failures})
    if failures:
        raise ValueError("Há imagens sem exatamente um rosto. Revise preprocessing.json; nenhuma foi descartada silenciosamente.")
    return np.stack(arrays)


def make_figures(history, metrics, model, sample, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(np.array(metrics["confusion_matrix"]), display_labels=[name[:1].upper() + name[1:] for name in metrics["labels"]]).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35)
    ax.set(xlabel="Identidade prevista", ylabel="Identidade real", title="Matriz de confusão · teste independente")
    fig.tight_layout()
    fig.savefig(destination / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Perda", "Acurácia")):
        epochs = range(1, len(history[metric]) + 1)
        axis.plot(epochs, history[metric], label="Treino")
        axis.plot(epochs, history["val_" + metric], label="Validação")
        axis.set(xlabel="Época", title=title)
        axis.legend()
    fig.tight_layout()
    fig.savefig(destination / "training_curves.png", dpi=160)
    plt.close(fig)

    base = model.get_layer("mobilenetv2_1.00_160")
    probe = tf.keras.Model(base.input, base.get_layer("Conv1_relu").output)
    maps = probe(sample[None].astype("float32") / 127.5 - 1, training=False).numpy()[0]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for channel, ax in enumerate(axes.flat):
        ax.imshow(maps[:, :, channel], cmap="viridis")
        ax.set_title(f"Canal {channel + 1}")
        ax.axis("off")
    fig.suptitle("Ativações reais da primeira convolução · não são explicações da decisão")
    fig.tight_layout()
    fig.savefig(destination / "activation_maps.png", dpi=160)
    plt.close(fig)
    from PIL import Image
    Image.fromarray(sample).save(destination / "activation_input.png")


def train(epochs=20, batch_size=16):
    labels, known = dataset_records()
    external = unknown_records()
    validate_duplicates(known + external)
    if set(labels) & {r["external_identity"] for r in external}:
        raise ValueError("Os identificadores das pessoas externas devem ser diferentes dos integrantes.")
    records = known + external
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    run_dir, run_report = ARTIFACTS / run_id, REPORTS / run_id
    run_dir.mkdir()
    run_report.mkdir()
    write_report(run_report / "manifest.json", records)
    images = prepare_records(records, run_report)

    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    model = build_model(len(labels))
    indices = {name: index for index, name in enumerate(labels)}

    def select(split, unknown=False):
        mask = [i for i, r in enumerate(records)
                if r["split"] == split and (r["person"] == "Desconhecido") == unknown]
        return images[mask], np.array([indices.get(records[i]["person"], -1) for i in mask])

    def dataset(x, y, training=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(len(x), seed=42)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        return ds.batch(batch_size).with_options(options).prefetch(1)

    train_x, train_y = select("train")
    val_x, val_y = select("validation")
    print("Treinando a saída da CNN; base convolucional congelada.", flush=True)
    history = model.fit(dataset(train_x, train_y, True),
                        validation_data=dataset(val_x, val_y), epochs=epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=5, restore_best_weights=True)],
                        shuffle=False, verbose=2)

    def predict(x):
        return np.concatenate([model(x[i:i + batch_size], training=False).numpy()
                               for i in range(0, len(x), batch_size)])

    unknown_val_x, _ = select("validation", True)
    calibration = calibrate(predict(val_x), val_y, predict(unknown_val_x), labels)
    # Somente agora o teste é usado. Ele não escolhe pesos, épocas ou limiar.
    test_x, test_y = select("test")
    unknown_test_x, _ = select("test", True)
    truth = [labels[int(i)] for i in test_y] + ["Desconhecido"] * len(unknown_test_x)
    metrics = report_metrics(predict(np.concatenate([test_x, unknown_test_x])), truth,
                             labels, calibration["threshold"])
    history_data = {k: [float(v) for v in values] for k, values in history.history.items()}
    make_figures(history_data, metrics, model, train_x[0], run_report)
    model.save(run_dir / "model.keras")
    metadata = {"run_id": run_id, "labels": labels, "threshold": calibration["threshold"],
                "architecture": "MobileNetV2 / ImageNet / base congelada",
                "image_size": 160, "seed": 42, "epochs_completed": len(history.epoch),
                "calibration": calibration, "created_at": datetime.now(timezone.utc).isoformat(),
                "tensorflow_version": tf.__version__}
    write_report(run_dir / "metadata.json", metadata)
    write_report(run_report / "evaluation.json", metrics)
    write_report(run_report / "history.json", history_data)
    # Só ativa uma execução completa; falhas preservam o último modelo válido.
    temp = ARTIFACTS / "active.tmp"
    write_report(temp, metadata)
    temp.replace(ARTIFACTS / "active.json")
    print(f"Concluído: {run_report}\nModelo disponível na interface. Limiar: {calibration['threshold']:.4f}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 200 or not 1 <= args.batch_size <= 128:
        parser.error("Use 1–200 épocas e lotes de 1–128 imagens.")
    try:
        train(args.epochs, args.batch_size)
    except ValueError as exc:
        parser.exit(2, f"Base ainda não pronta: {exc}\n")


if __name__ == "__main__":
    main()
