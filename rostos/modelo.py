"""MobileNetV2 original e carregamento das CNNs de reconhecimento versionadas."""
import json

import numpy as np

from .config import ARTIFACTS, IMAGE_SIZE


def build_model(class_count: int, weights="imagenet"):
    import tensorflow as tf
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights=weights)
    base.trainable = False
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="rosto_rgb")
    # Aumentos só atuam durante fit(training=True), nunca em validação ou teste.
    x = tf.keras.layers.RandomFlip("horizontal", seed=42)(inputs)
    x = tf.keras.layers.RandomRotation(0.035, fill_mode="reflect", seed=43)(x)
    x = tf.keras.layers.RandomContrast(0.12, seed=44)(x)
    # Incluída no modelo salvo: pixels [0, 255] passam para [-1, 1].
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="normalizacao")(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="resumo_espacial")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="integrantes")(x)
    model = tf.keras.Model(inputs, outputs, name="cnn_integrantes")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def decisions(probabilities, labels, threshold):
    probabilities = np.asarray(probabilities)
    indices = probabilities.argmax(axis=1)
    scores = probabilities.max(axis=1)
    return [{"label": labels[int(index)] if float(score) >= threshold else "Desconhecido",
             "score": float(score), "accepted": bool(float(score) >= threshold)}
            for index, score in zip(indices, scores)]


def active_metadata():
    pointer = ARTIFACTS / "active.json"
    if not pointer.exists():
        return None
    metadata = json.loads(pointer.read_text())
    path = artifact_path(metadata)
    if not path.is_relative_to(ARTIFACTS.resolve()) or not path.is_file():
        raise ValueError("O modelo ativo não está disponível. Execute um treinamento completo.")
    return metadata


def artifact_path(metadata):
    model_format = metadata.get("model_format", "keras")
    filenames = {"keras": "model.keras", "torch_facenet_v1": "model.pt"}
    if model_format not in filenames:
        raise ValueError("Formato de modelo não suportado.")
    path = (ARTIFACTS / metadata["run_id"] / filenames[model_format]).resolve()
    if not path.is_relative_to(ARTIFACTS.resolve()):
        raise ValueError("Caminho de modelo inválido.")
    return path


def load_network(metadata):
    if metadata.get("model_format") == "torch_facenet_v1":
        from .treinamento_facial import load_model
        return load_model(artifact_path(metadata), len(metadata["labels"]))
    import tensorflow as tf
    return tf.keras.models.load_model(artifact_path(metadata), compile=False)


def network_scores(network, batch, metadata):
    batch = np.asarray(batch, dtype="float32")
    if metadata.get("model_format") == "torch_facenet_v1":
        return network.predict_numpy(batch)
    return network(batch, training=False).numpy()


_ACTIVE_MODEL = object()


class Recognizer:
    def __init__(self):
        self.run_id = None
        self.network = None

    def predict(self, batch, metadata=_ACTIVE_MODEL):
        if metadata is _ACTIVE_MODEL:
            metadata = active_metadata()
        if metadata is None:
            return [{"label": "Reconhecimento ainda não treinado", "score": None,
                     "accepted": False} for _ in batch]
        if self.run_id != metadata["run_id"]:
            self.network = load_network(metadata)
            self.run_id = metadata["run_id"]
        maps, feature_count = None, None
        if metadata.get("model_format") == "torch_facenet_v1":
            probabilities, maps, feature_count = self.network.inspect_numpy(batch)
        else:
            probabilities = network_scores(self.network, batch, metadata)
        results = decisions(probabilities, metadata["labels"], metadata["threshold"])
        for index, result in enumerate(results):
            result.update(scores_by_class=dict(zip(metadata["labels"], map(float, probabilities[index]))),
                          threshold=metadata["threshold"], feature_count=feature_count,
                          activation_maps=[])
            if maps is not None:
                from PIL import Image
                from .imagem import image_url
                for channel, values in enumerate(maps[index]):
                    low, high = float(values.min()), float(values.max())
                    pixels = (255 * (values - low) / (high - low)).astype("uint8") if high > low else np.zeros_like(values, dtype="uint8")
                    result["activation_maps"].append({"channel": channel + 1,
                        "image": image_url(Image.fromarray(pixels).convert("RGB")),
                        "min": low, "max": high, "size": list(values.shape)})
        return results
