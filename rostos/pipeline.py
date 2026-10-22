"""Fluxo principal: ler → detectar → recortar → padronizar → CNN → decisão → relatório.

Comece por _analyze para estudar a ordem. analyze protege as redes compartilhadas
por sessões simultâneas; não guarda fotos ou resultados em variáveis globais.
"""
from threading import RLock
import numpy as np
from .imagem import read_image, detect, image_url
from .modelo import Recognizer, active_metadata
from .etapas.preparacao import prepare_faces, describe_input
from .saida import build_report, model_mode

recognizer = Recognizer()
processing_lock = RLock()

def analyze(content, filename="imagem"):
    with processing_lock:
        return _analyze(content, filename)

def _analyze(content, filename="imagem"):
    # Uma única versão para previsão, limiar e relatório, mesmo se outro
    # treinamento terminar enquanto esta imagem é processada.
    metadata = active_metadata()
    image = read_image(content)
    detections = detect(image)
    crops, inputs = prepare_faces(image, detections)
    predictions = recognizer.predict([np.asarray(img) for img in inputs], metadata=metadata) if inputs else []
    faces = []
    for i, (detection, crop, processed, prediction) in enumerate(zip(detections, crops, inputs, predictions)):
        faces.append({"id": i + 1, **detection, **prediction,
                      **describe_input(crop, processed, metadata)})
    model_info = {key: metadata.get(key) for key in
                  ("run_id", "architecture", "threshold", "threshold_calibrated", "model_format",
                   "preprocessing", "normalization", "labels", "training_images", "experimental")} if metadata else None
    report = build_report(content, filename, image, faces, model_info, metadata)
    return {"image": image_url(image), "width": image.width, "height": image.height,
            "faces": faces, "mode": model_mode(metadata),
            "model": model_info, "report": report}

