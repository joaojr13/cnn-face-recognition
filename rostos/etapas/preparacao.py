"""Etapas 3–5: mesmos pixels e mesma normalização do treinamento."""
import hashlib
import numpy as np
from ..imagem import crop_face, model_image, image_url


def prepare_faces(image, detections):
    crops = [crop_face(image, detection["box"]) for detection in detections]
    return crops, [model_image(crop) for crop in crops]


def describe_input(crop, processed, metadata):
    pixels = np.asarray(processed, dtype="float32")
    divisor = 128.0 if metadata and metadata.get("model_format") == "torch_facenet_v1" else 127.5
    normalized = (pixels - 127.5) / divisor
    return {"crop": image_url(crop), "processed": image_url(processed),
            "input_size": list(processed.size),
            "input_sha256": hashlib.sha256(np.asarray(processed).tobytes()).hexdigest(),
            "normalization_example": {"position": [80, 80], "rgb": pixels[80, 80].tolist(),
                "normalized_rgb": normalized[80, 80].tolist(),
                "min": float(normalized.min()), "max": float(normalized.max())},
            "normalization": metadata.get("normalization", "pixel / 127,5 − 1") if metadata else "pixel / 127,5 − 1"}
