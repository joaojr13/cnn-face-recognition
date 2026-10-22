"""Registro auditável de uma previsão, sem imagens nem caminhos absolutos."""
import hashlib
from datetime import datetime, timezone
import numpy as np

def model_mode(metadata):
    if not metadata:
        return "detection_only"
    return "experimental" if metadata.get("experimental") else "recognition"

def build_report(content, filename, image, faces, model_info, metadata):
    report = {"schema_version": 1, "analyzed_at": datetime.now(timezone.utc).isoformat(),
              "purpose": "Diagnóstico de uma imagem; não é avaliação independente nem confirmação de identidade.",
              "source": {"filename": filename.replace("\\", "/").split("/")[-1],
                         "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                         "processed_size": [image.width, image.height],
                         "processed_rgb_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest()},
              "pipeline": "mtcnn_pnet_min12_crop8_pad160_v2", "model": model_info,
              "mode": model_mode(metadata), "faces": []}
    for face in faces:
        row = {key: face.get(key) for key in ("id", "box", "detection_score", "label", "score", "accepted",
               "scores_by_class", "threshold", "input_sha256", "normalization", "normalization_example")}
        ranked = sorted((face.get("scores_by_class") or {}).items(), key=lambda item: item[1], reverse=True)
        row["top_candidate"] = ranked[0][0] if ranked else None
        row["decision_reason"] = ("recognizer_not_trained" if face["score"] is None else
                                  "threshold_met" if face["accepted"] else "below_threshold")
        row["margin_percentage_points"] = (100 * (face["score"] - metadata["threshold"])
                                              if face["score"] is not None and metadata else None)
        report["faces"].append(row)
    return report
