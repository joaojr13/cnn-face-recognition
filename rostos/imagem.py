"""O mesmo tratamento é usado no treinamento e no reconhecimento."""
import base64
import io
import warnings
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import IMAGE_SIZE, MAX_PIXELS, MAX_SIDE, MAX_UPLOAD_BYTES

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def read_image(content: bytes) -> Image.Image:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Envie uma imagem de até 12 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as original:
                if original.format not in {"JPEG", "MPO", "PNG", "WEBP"}:
                    raise ValueError("Use uma imagem JPEG, PNG ou WebP.")
                # Alguns arquivos .jpg de celulares contêm imagens auxiliares
                # (MPO). Só a primeira imagem é a foto usada no reconhecimento.
                original.seek(0)
                if original.width * original.height > MAX_PIXELS:
                    raise ValueError("A imagem deve ter no máximo 24 megapixels.")
                result = ImageOps.exif_transpose(original).convert("RGB")
                result.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
                return result.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("Não foi possível abrir a imagem. Use JPEG, PNG ou WebP válido.") from exc


def image_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@lru_cache(maxsize=1)
def detector():
    from mtcnn import MTCNN
    from mtcnn.stages import StageRNet, StageONet
    from .deteccao import SafeStagePNet
    # A detecção em CPU é suficiente para uploads e evita depender do Metal.
    return MTCNN(stages=[SafeStagePNet, StageRNet, StageONet], device="CPU:0")


def detect(image: Image.Image) -> list[dict]:
    detections = detector().detect_faces(np.asarray(image), min_face_size=24)
    faces = []
    for item in detections:
        if float(item["confidence"]) < 0.95:
            continue
        x, y, width, height = (int(v) for v in item["box"])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.width, x + width), min(image.height, y + height)
        if x2 - x1 < 16 or y2 - y1 < 16:
            continue
        faces.append({"box": [x1, y1, x2, y2],
                      "detection_score": float(item["confidence"])})
    return sorted(faces, key=lambda face: (face["box"][0], face["box"][1]))


def crop_face(image: Image.Image, box: list[int]) -> Image.Image:
    # Margem proporcional ao rosto; a mesma regra vale para treino e upload.
    x1, y1, x2, y2 = box
    margin = round(max(x2 - x1, y2 - y1) * 0.08)
    return image.crop((max(0, x1 - margin), max(0, y1 - margin),
                       min(image.width, x2 + margin), min(image.height, y2 + margin)))


def model_image(crop: Image.Image) -> Image.Image:
    # Preserva proporções; preenchimento preto evita deformar o rosto.
    return ImageOps.pad(crop, (IMAGE_SIZE, IMAGE_SIZE),
                        method=Image.Resampling.LANCZOS, color=(0, 0, 0))
