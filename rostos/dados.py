"""Inventário e separação por sessão, antes de qualquer aumento de dados."""
import hashlib
from pathlib import Path

from .config import DATA, EXPECTED_PEOPLE, EXTENSIONS


def images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)


def inventory(root: Path = DATA / "raw") -> dict:
    people = []
    issues = []
    for person in sorted(p for p in root.iterdir() if p.is_dir()):
        sessions = [{"name": session.name, "count": len(images_in(session))}
                    for session in sorted(person.iterdir()) if session.is_dir()]
        sessions = [s for s in sessions if s["count"]]
        count = len(images_in(person))
        people.append({"name": person.name, "count": count, "sessions": sessions})
        if len(sessions) < 3:
            issues.append(f"{person.name}: faltam sessões independentes (mínimo de 3).")
        if any(p.is_file() and p.suffix.lower() in EXTENSIONS for p in person.iterdir()):
            issues.append(f"{person.name}: organize as fotos dentro de pastas de sessão.")
    if len(people) != EXPECTED_PEOPLE:
        issues.insert(0, f"Há {len(people)} de {EXPECTED_PEOPLE} integrantes na base.")
    return {"people": people, "total_images": sum(p["count"] for p in people),
            "expected_people": EXPECTED_PEOPLE, "issues": issues}


def dataset_records(root: Path = DATA / "raw", expected: int = EXPECTED_PEOPLE) -> tuple[list, list]:
    people = sorted(p for p in root.iterdir() if p.is_dir())
    if len(people) != expected:
        raise ValueError(f"São necessárias pastas para os {expected} integrantes; encontradas {len(people)}.")
    if any(p.name.casefold() == "desconhecido" for p in people):
        raise ValueError("Desconhecido é um nome reservado para rejeições, não para integrantes.")
    records = []
    for person in people:
        sessions = sorted(p for p in person.iterdir() if p.is_dir() and images_in(p))
        if any(p.is_file() and p.suffix.lower() in EXTENSIONS for p in person.iterdir()):
            raise ValueError(f"{person.name}: coloque cada foto em uma pasta de sessão.")
        if len(sessions) < 3:
            raise ValueError(f"{person.name}: forneça ao menos 3 sessões independentes.")
        for i, session in enumerate(sessions):
            split = "test" if i == len(sessions) - 1 else "validation" if i == len(sessions) - 2 else "train"
            for path in images_in(session):
                records.append({"path": str(path), "person": person.name,
                                "session": session.name, "split": split})
        for split, minimum in (("train", 10), ("validation", 3), ("test", 3)):
            count = sum(r["person"] == person.name and r["split"] == split for r in records)
            if count < minimum:
                raise ValueError(f"{person.name}: {split} tem {count} fotos; mínimo operacional: {minimum}.")
    validate_duplicates(records)
    return [p.name for p in people], records


def unknown_records(root: Path = DATA / "unknown") -> list:
    result = []
    identities = {}
    for split in ("validation", "test"):
        folder = root / split
        people = sorted(p for p in folder.iterdir() if p.is_dir()) if folder.exists() else []
        identities[split] = {p.name for p in people if images_in(p)}
        if not identities[split]:
            raise ValueError(f"Adicione pessoas externas em data/unknown/{split}/pessoa/.")
        for person in people:
            for path in images_in(person):
                result.append({"path": str(path), "person": "Desconhecido",
                               "external_identity": person.name, "split": split, "session": person.name})
    if identities["validation"] & identities["test"]:
        raise ValueError("Pessoas externas da validação e do teste devem ser diferentes (use identificadores únicos).")
    validate_duplicates(result)
    return result


def validate_duplicates(records: list) -> None:
    seen = {}
    for record in records:
        digest = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(f"Arquivo duplicado: {record['path']} e {seen[digest]}. Remova a duplicata da base.")
        seen[digest] = record["path"]
        record["sha256"] = digest
