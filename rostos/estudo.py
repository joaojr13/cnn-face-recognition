"""Evidências didáticas da execução ativa; nenhum treinamento é iniciado aqui."""
import hashlib
import json
from pathlib import Path

from .config import DATA, REPORTS
from .arquivos import resolve_record_path
from .imagem import crop_face, detect, image_url, model_image, read_image


def read_report(metadata, filename, default):
    path = REPORTS / metadata["run_id"] / filename
    return json.loads(path.read_text()) if path.is_file() else default


def training_records(metadata):
    return [record for record in read_report(metadata, "manifest.json", [])
            if record["split"] in {"train", "experimental_train_only"}]


def summarize_training(predictions, labels):
    """Matriz com rejeição, médias somente das identidades presentes no treino."""
    names = labels + ["Desconhecido"]
    matrix = [[0 for _ in names] for _ in names]
    for prediction in predictions:
        matrix[names.index(prediction["training_label"])][names.index(prediction["label"])] += 1
    rows = []
    for index, name in enumerate(names):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": name, "support": support,
                     "precision": precision if support else None,
                     "recall": recall if support else None, "f1": f1 if support else None})
    supported = [row for row in rows if row["support"]]
    total = len(predictions)
    return {"scope": "training_only", "independent": False, "labels": names,
            "matrix": matrix, "rows": rows, "size": total,
            "accuracy": sum(matrix[i][i] for i in range(len(names))) / total if total else None,
            "macro_precision": sum(row["precision"] for row in supported) / len(supported) if supported else None,
            "macro_recall": sum(row["recall"] for row in supported) / len(supported) if supported else None,
            "macro_f1": sum(row["f1"] for row in supported) / len(supported) if supported else None,
            "unknown_false_accept_rate": None}


def study_summary(metadata):
    records = training_records(metadata)
    diagnostics = read_report(metadata, "training_diagnostics.json", {}).get("predictions", [])
    # Relaciona as previsões ao manifesto desta execução, não à pasta atual.
    labels_by_path = {record["path"]: record["person"] for record in records}
    valid = bool(diagnostics) and len(diagnostics) == len(records) and len({p["path"] for p in diagnostics}) == len(records)
    valid = valid and all(labels_by_path.get(p["path"]) == p["training_label"] for p in diagnostics)
    return {"available": True, "metadata": metadata,
            "training_images": [{"id": index, "person": record["person"],
                "filename": Path(record["path"]).name, "session": record["session"],
                "sha256": record.get("sha256"), "split": record["split"]}
                for index, record in enumerate(records)],
            "history": read_report(metadata, "history.json", {}),
            "training_diagnostic": summarize_training(diagnostics, metadata["labels"]) if valid else None}


def training_image(metadata, index):
    records = training_records(metadata)
    if not 0 <= index < len(records):
        raise IndexError("Imagem não encontrada.")
    record = records[index]
    path = resolve_record_path(record["path"])
    if not path.is_relative_to((DATA / "raw").resolve()):
        raise ValueError("A imagem não pertence à base local de treinamento.")
    content = path.read_bytes()
    if not record.get("sha256") or hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise ValueError("O arquivo mudou desde o treinamento; a reprodução foi bloqueada.")
    original = read_image(content)
    box = record.get("box")
    if box is None:
        detections = detect(original)
        if len(detections) != 1:
            raise ValueError("Não foi possível reproduzir o recorte desta execução.")
        box = detections[0]["box"]
    crop = crop_face(original, box)
    processed = model_image(crop)
    variants = []
    if metadata.get("model_format") == "torch_facenet_v1":
        import numpy as np
        from PIL import Image
        from .treinamento_facial import training_variants
        names = ["Original padronizada", "Espelho", "Tons de cinza", "Rotação −4°", "Rotação +4°",
                 "Brilho −15%", "Brilho +15%", "Contraste −15%", "Contraste +15%"]
        variants = [{"name": name, "image": image_url(Image.fromarray(pixels))}
                    for name, pixels in zip(names, training_variants(np.asarray(processed)))]
    return {"person": record["person"], "filename": path.name, "sha256": record["sha256"],
            "original": image_url(original), "crop": image_url(crop), "processed": image_url(processed),
            "box": box, "variants": variants}


def evaluation(metadata):
    if not metadata:
        return {"available": False}
    if metadata.get("experimental"):
        return {"available": False, "experimental": True, "metadata": metadata}
    metrics = read_report(metadata, "evaluation.json", None)
    return {"available": metrics is not None, "metadata": metadata, "metrics": metrics}
