"""Avaliação do modelo facial ativo com as fotos já existentes, sem treinar a base.

Protocolo (nenhuma foto nova é necessária):

1. Escolha do limiar (validação) - o teste não participa:
   - integrantes: leave-one-out nas 10 fotos de treino. Cada foto é prevista por
     uma camada final treinada do zero sem ela, com o mesmo procedimento do modelo ativo;
   - externos: data/unknown/validation (25 pessoas LFW), pontuadas pelo modelo ativo.
   O limiar maximiza o F1 macro com no máximo 5% de externos aceitos.
2. Teste final: o modelo ativo, sem alterações, prevê data/test/<integrante>
   (fotos nunca usadas) e data/unknown/test (outras 25 pessoas LFW).
3. Validação cruzada complementar: leave-one-out com as fotos de treino e de teste
   dos integrantes (3 por pessoa), usando o limiar já escolhido. Dá mais exemplos
   por pessoa para a matriz, mas não substitui o teste do modelo ativo.

Uso: .venv/bin/python -m rostos.avaliacao_facial [--aplicar-limiar]
"""
import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone

import numpy as np

from .arquivos import resolve_record_path, write_report
from .config import ARTIFACTS, DATA, REPORTS, ROOT
from .dados import images_in, validate_duplicates
from .treinamento_mobilenet import calibrate, report_metrics

UNKNOWN = "Desconhecido"
DEFAULT_MAX_FAR = 0.05
FIGURES = {"test": "matriz_confusao_teste.png",
           "cross_validation": "matriz_confusao_validacao_cruzada.png",
           "scores": "distribuicao_pontuacoes.png"}
PROTOCOL = [
    "Limiar escolhido na validação: leave-one-out nas fotos de treino dos integrantes "
    "+ 50 fotos LFW de validação (25 pessoas). O teste não participa da escolha.",
    "Teste final: modelo ativo sem alterações em 1 foto nova por integrante "
    "+ 50 fotos LFW de teste (outras 25 pessoas).",
    "Validação cruzada complementar: leave-one-out com as 3 fotos de cada integrante "
    "(treino + teste); cada foto é prevista por uma camada final treinada sem ela.",
]


# ---------------------------------------------------------------- dados ----

def relative(path):
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def member_test_records(labels, root=DATA / "test"):
    """Fotos novas dos integrantes em data/test/<integrante>/ (nunca usadas no treino)."""
    folders = {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()
    unexpected = sorted(folders - set(labels))
    if unexpected:
        raise ValueError(f"Pastas de teste sem integrante correspondente: {', '.join(unexpected)}.")
    records = []
    for label in labels:
        photos = images_in(root / label) if (root / label).exists() else []
        if not photos:
            raise ValueError(f"{label}: coloque ao menos uma foto nova em data/test/{label}/.")
        records += [{"path": relative(path), "person": label, "split": "test", "session": "test"}
                    for path in photos]
    validate_duplicates(records)
    return records


def external_records(root=DATA / "unknown"):
    """Base LFW conferida pelo manifesto (hash, divisão e caixa do rosto)."""
    from .desconhecidos import unknown_summary
    if not unknown_summary(root).get("available"):
        raise ValueError("A base externa não está disponível. Execute scripts/prepare_unknown_faces.py.")
    records = json.loads((root / "manifest.json").read_text())["records"]
    return ([r for r in records if r["split"] == "validation"],
            [r for r in records if r["split"] == "test"])


def check_no_overlap(train, test):
    shared = {r["sha256"] for r in train} & {r["sha256"] for r in test}
    if shared:
        raise ValueError("Uma foto de teste é idêntica a uma foto de treino. Remova a duplicata.")


def face_inputs(records):
    """Mesma preparação do treino: leitura → caixa → recorte 8% → 160 × 160.

    Usa a caixa registrada quando existe (treino e LFW); detecta nas fotos novas.
    """
    from .imagem import crop_face, detect, model_image, read_image
    arrays, failures = [], []
    for record in records:
        try:
            content = resolve_record_path(record["path"]).read_bytes()
            if record.get("sha256") and hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError("o arquivo mudou desde o registro")
            image = read_image(content)
            box = record.get("box")
            if box is None:
                faces = detect(image)
                if len(faces) != 1:
                    raise ValueError(f"esperado exatamente 1 rosto; encontrados {len(faces)}")
                box = faces[0]["box"]
                record["box"], record["detection_score"] = box, faces[0]["detection_score"]
            arrays.append(np.asarray(model_image(crop_face(image, box))))
        except (OSError, ValueError) as exc:
            failures.append(f"{record['path']}: {exc}")
    if failures:
        raise ValueError("Imagens sem preparação válida:\n" + "\n".join(failures))
    return np.stack(arrays)


# ------------------------------------------------------------- protocolo ----

def leave_one_out_probabilities(augmented, targets, fit, predict):
    """Cada foto é prevista por uma camada treinada sem ela (nem suas variações).

    augmented: (fotos, variações, características); a variação 0 é a foto original.
    """
    augmented, targets = np.asarray(augmented), np.asarray(targets)
    count, variants, size = augmented.shape
    rows = []
    for index in range(count):
        keep = np.arange(count) != index
        head = fit(augmented[keep].reshape(-1, size), np.repeat(targets[keep], variants))
        rows.append(predict(head, augmented[index, :1])[0])
    return np.stack(rows)


def metrics_block(probabilities, truth, labels, threshold, paths):
    probabilities = np.asarray(probabilities, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # média vazia quando não há externos
        metrics = report_metrics(probabilities, truth, labels, threshold)
    known = np.array([name != UNKNOWN for name in truth])
    if known.all():
        metrics["unknown_false_accept_rate"] = None
    # Médias macro só com as classes que têm exemplos: sem externos, a coluna
    # Desconhecido (rejeições) não entra como uma classe de F1 zero.
    rows = metrics["classification_report"]
    supported = [name for name in labels + [UNKNOWN] if rows[name]["support"] > 0]
    for key, field in (("macro_precision", "precision"), ("macro_recall", "recall"), ("macro_f1", "f1-score")):
        metrics[key] = float(np.mean([rows[name][field] for name in supported]))
    metrics["macro_classes"] = supported
    metrics["threshold"] = float(threshold)
    metrics["known_images"] = int(known.sum())
    metrics["unknown_images"] = int((~known).sum())
    metrics["samples"] = [
        {"path": path, "truth": true, "prediction": predicted,
         "top_candidate": labels[int(row.argmax())], "score": float(row.max())}
        for path, true, predicted, row in zip(paths, truth, metrics["predictions"], probabilities)]
    return metrics


def run_protocol(labels, train, member_test, unknown_validation, unknown_test,
                 fit, predict, provisional_threshold, max_far=DEFAULT_MAX_FAR):
    """Executa validação → teste → validação cruzada a partir de características prontas.

    train / member_test: {"augmented", "targets", "paths"}; member_test também tem
    "probabilities" do modelo ativo. unknown_*: {"probabilities", "paths"}.
    fit(características, alvos) → camada; predict(camada, características) → softmax.
    """
    names = [labels[int(i)] for i in train["targets"]]
    known_validation = leave_one_out_probabilities(train["augmented"], train["targets"], fit, predict)
    calibration = calibrate(known_validation, train["targets"],
                            np.asarray(unknown_validation["probabilities"]), labels, max_far)
    threshold = calibration["threshold"]
    calibration.update(
        method="F1 macro máximo com taxa de falsos aceites de externos ≤ alvo (validação).",
        known_validation_images=len(names),
        unknown_validation_images=len(unknown_validation["paths"]))

    validation = metrics_block(
        np.concatenate([known_validation, unknown_validation["probabilities"]]),
        names + [UNKNOWN] * len(unknown_validation["paths"]), labels, threshold,
        list(train["paths"]) + list(unknown_validation["paths"]))

    test_probabilities = np.concatenate([member_test["probabilities"], unknown_test["probabilities"]])
    test_truth = [labels[int(i)] for i in member_test["targets"]] + [UNKNOWN] * len(unknown_test["paths"])
    test_paths = list(member_test["paths"]) + list(unknown_test["paths"])
    test = metrics_block(test_probabilities, test_truth, labels, threshold, test_paths)
    test_provisional = metrics_block(test_probabilities, test_truth, labels, provisional_threshold, test_paths)

    all_augmented = np.concatenate([train["augmented"], member_test["augmented"]])
    all_targets = np.concatenate([train["targets"], member_test["targets"]])
    cross = metrics_block(
        leave_one_out_probabilities(all_augmented, all_targets, fit, predict),
        [labels[int(i)] for i in all_targets], labels, threshold,
        list(train["paths"]) + list(member_test["paths"]))

    return {"scope": "independent_test", "independent": True, "protocol": PROTOCOL,
            "labels": labels + [UNKNOWN], "threshold": float(threshold),
            "provisional_threshold": float(provisional_threshold), "calibration": calibration,
            "validation": validation, "test": test,
            "test_at_provisional_threshold": test_provisional, "cross_validation": cross,
            "limitations": [
                f"O teste tem {len(member_test['paths'])} fotos de integrantes: cada recall por pessoa só pode ser 0% ou 100%.",
                "A validação cruzada reutiliza as fotos de teste como exemplos previstos; é complementar.",
                "O limiar foi escolhido com camadas treinadas com uma foto a menos que o modelo ativo.",
                "As fotos LFW não representam a câmera ou a iluminação das fotos do grupo."]}


def threshold_effect(samples, labels, threshold):
    """Reaplica um limiar às pontuações salvas, sem executar a rede.

    Usa a mesma regra de modelo.decisions e o mesmo F1 macro de calibrate().
    """
    from sklearn.metrics import f1_score
    truth = [s["truth"] for s in samples]
    predicted = [s["top_candidate"] if s["score"] >= threshold else UNKNOWN for s in samples]
    members = [(t, p) for t, p in zip(truth, predicted) if t != UNKNOWN]
    outsiders = [p for t, p in zip(truth, predicted) if t == UNKNOWN]
    accepted_outsiders = sum(p != UNKNOWN for p in outsiders)
    return {"threshold": float(threshold),
            "members": len(members), "outsiders": len(outsiders),
            "members_correct": sum(t == p for t, p in members),
            "members_rejected": sum(p == UNKNOWN for _, p in members),
            "members_wrong_name": sum(p not in (t, UNKNOWN) for t, p in members),
            "outsiders_accepted": accepted_outsiders,
            "false_accept_rate": accepted_outsiders / len(outsiders) if outsiders else None,
            "rejection_rate": (sum(p == UNKNOWN for _, p in members) / len(members)) if members else None,
            "macro_f1": float(f1_score(truth, predicted, labels=labels + [UNKNOWN],
                                       average="macro", zero_division=0))}


def threshold_candidates(result):
    """Limiares representativos para explicar a escolha, com uma observação cada."""
    validation = result["validation"]["samples"]
    members = [s["score"] for s in validation if s["truth"] != UNKNOWN]
    outsiders = [s["score"] for s in validation if s["truth"] == UNKNOWN]
    notes = [(0.80, ""), (0.99, ""), (result["provisional_threshold"], "provisório anterior"),
             (min(members), "menor pontuação de integrante: aceita todos"),
             (max(outsiders) + 1e-7, "logo acima do externo mais alto: rejeita todos"),
             (result["threshold"], "calibrado (escolhido)")]
    rows = {}
    for value, note in notes:
        key = round(value, 4)
        rows[key] = (value, "; ".join(filter(None, [rows.get(key, (0, ""))[1], note])))
    return [(value, note) for _, (value, note) in sorted(rows.items())]


# --------------------------------------------------------------- saídas ----

def save_figures(result, destination):
    from .graficos import confusion_figure, save, score_figure
    save(confusion_figure(result["labels"], result["test"]["confusion_matrix"],
                          "Matriz de confusão · teste independente"), destination / FIGURES["test"])
    save(confusion_figure(result["labels"], result["cross_validation"]["confusion_matrix"],
                          "Matriz de confusão · validação cruzada (integrantes)"),
         destination / FIGURES["cross_validation"])
    save(score_figure(score_panels(result), result["threshold"], result["provisional_threshold"]),
         destination / FIGURES["scores"])
    return dict(FIGURES)


def score_panels(result):
    def split(block):
        samples = block["samples"]
        return ([s["score"] for s in samples if s["truth"] != UNKNOWN],
                [s["score"] for s in samples if s["truth"] == UNKNOWN])
    return [("Validação (escolha do limiar)", *split(result["validation"])),
            ("Teste independente", *split(result["test"]))]


def apply_threshold(run_id, result, artifacts=ARTIFACTS):
    """Grava o limiar calibrado no modelo avaliado; os pesos não mudam."""
    path = artifacts / run_id / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.setdefault("provisional_threshold", metadata["threshold"])
    test = result["test"]
    metadata.update(
        threshold=result["threshold"], threshold_calibrated=True, evaluation_available=True,
        calibration={key: result["calibration"][key] for key in
                     ("threshold", "target_max_far", "validation_macro_f1",
                      "validation_unknown_false_accept_rate", "known_validation_images",
                      "unknown_validation_images", "method")},
        limitations=(f"Avaliado com teste pequeno: {test['known_images']} fotos novas de integrantes "
                     f"e {test['unknown_images']} fotos externas. Poucas fotos de treino."))
    write_report(path, metadata)
    pointer = artifacts / "active.json"
    if pointer.is_file() and json.loads(pointer.read_text()).get("run_id") == run_id:
        temporary = artifacts / f"{run_id}.tmp"
        write_report(temporary, metadata)
        temporary.replace(pointer)
    return metadata


def evaluate(metadata, apply=False, max_far=DEFAULT_MAX_FAR):
    if not metadata or metadata.get("model_format") != "torch_facenet_v1":
        raise ValueError("A avaliação facial exige o modelo FaceNet ativo.")
    from .estudo import training_records
    from .modelo import load_network
    from .treinamento_facial import (backbone_features, fit_head, head_probabilities,
                                     training_variants)
    labels = metadata["labels"]
    train = training_records(metadata)
    test = member_test_records(labels)
    check_no_overlap(train, test)
    unknown_validation, unknown_test = external_records()
    print(f"Fotos: {len(train)} treino · {len(test)} teste de integrantes · "
          f"{len(unknown_validation)} externos validação · {len(unknown_test)} externos teste", flush=True)

    model = load_network(metadata)
    print("Preparando rostos (mesmo recorte e padronização do treino)…", flush=True)
    images = {name: face_inputs(records) for name, records in
              [("train", train), ("test", test), ("uval", unknown_validation), ("utest", unknown_test)]}

    def augmented(batch):
        variants = np.concatenate([training_variants(image) for image in batch])
        features = backbone_features(model, variants)
        return features.reshape(len(batch), -1, features.shape[1])

    print("Calculando características da CNN facial…", flush=True)
    target = {label: index for index, label in enumerate(labels)}
    epochs = int(metadata.get("epochs_completed", 200))
    result = run_protocol(
        labels,
        train={"augmented": augmented(images["train"]), "paths": [r["path"] for r in train],
               "targets": np.array([target[r["person"]] for r in train])},
        member_test={"augmented": augmented(images["test"]), "paths": [r["path"] for r in test],
                     "targets": np.array([target[r["person"]] for r in test]),
                     "probabilities": model.predict_numpy(images["test"])},
        unknown_validation={"probabilities": model.predict_numpy(images["uval"]),
                            "paths": [r["path"] for r in unknown_validation]},
        unknown_test={"probabilities": model.predict_numpy(images["utest"]),
                      "paths": [r["path"] for r in unknown_test]},
        fit=lambda features, targets: fit_head(features, targets, len(labels), epochs),
        predict=head_probabilities,
        provisional_threshold=metadata.get("provisional_threshold", metadata["threshold"]),
        max_far=max_far)
    result.update(run_id=metadata["run_id"], created_at=datetime.now(timezone.utc).isoformat(),
                  head_epochs=epochs,
                  member_test_records=[{key: r.get(key) for key in
                                        ("path", "person", "sha256", "box", "detection_score")}
                                       for r in test])
    destination = REPORTS / metadata["run_id"]
    destination.mkdir(parents=True, exist_ok=True)
    result["figures"] = save_figures(result, destination)
    write_report(destination / "evaluation.json", result)
    if apply:
        apply_threshold(metadata["run_id"], result)
    return result


def print_summary(result, applied):
    test, cross = result["test"], result["cross_validation"]
    far = test["unknown_false_accept_rate"]
    print("\n=== Limiar ===")
    print(f"Calibrado: {result['threshold']:.4f} (provisório anterior: {result['provisional_threshold']:.2f})")
    print("\n=== Teste independente ===")
    print(f"Acurácia: {test['accuracy']:.1%} · precisão macro {test['macro_precision']:.3f} · "
          f"recall macro {test['macro_recall']:.3f} · F1 macro {test['macro_f1']:.3f}")
    print(f"Integrantes rejeitados: {test['known_rejection_rate']:.1%} · "
          f"externos aceitos: {far:.1%}")
    for sample in test["samples"][:test["known_images"]]:
        print(f"  {sample['truth']:>10} → {sample['prediction']:<12} "
              f"(candidato {sample['top_candidate']}, {sample['score']:.2%})")
    print("\n=== Validação cruzada (complementar) ===")
    print(f"Acurácia: {cross['accuracy']:.1%} em {cross['known_images']} fotos · "
          f"F1 macro {cross['macro_f1']:.3f} · rejeitados {cross['known_rejection_rate']:.1%}")
    print(f"\nRelatório: reports/{result['run_id']}/evaluation.json e figuras PNG na mesma pasta.")
    print("Limiar calibrado gravado no modelo ativo." if applied else
          "O modelo ativo continua com o limiar anterior. Use --aplicar-limiar para gravar o calibrado.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aplicar-limiar", action="store_true",
                        help="Grava o limiar calibrado no modelo ativo (os pesos não mudam).")
    parser.add_argument("--max-far", type=float, default=DEFAULT_MAX_FAR,
                        help="Máximo de externos aceitos na validação (padrão 0,05).")
    args = parser.parse_args()
    if not 0 <= args.max_far <= 0.5:
        parser.error("Use --max-far entre 0 e 0,5.")
    from .modelo import active_metadata
    try:
        result = evaluate(active_metadata(), args.aplicar_limiar, args.max_far)
    except ValueError as exc:
        parser.exit(2, f"Avaliação não concluída: {exc}\n")
    print_summary(result, args.aplicar_limiar)


if __name__ == "__main__":
    main()
