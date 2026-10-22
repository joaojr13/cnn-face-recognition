"""Copia imagens individuais para uma sessão, sem alterar os originais."""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rostos.config import DATA, EXTENSIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pasta com uma subpasta por integrante")
    parser.add_argument("--session", required=True, help="Ex.: sessao_01")
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error("A pasta de origem não existe.")
    if not args.session or args.session in {".", ".."} or Path(args.session).name != args.session:
        parser.error("Use um nome simples para a sessão, sem barras.")
    copied, skipped = 0, 0
    for person in sorted(p for p in args.source.iterdir() if p.is_dir()):
        for source in sorted(person.iterdir()):
            if not source.is_file() or source.suffix.lower() not in EXTENSIONS:
                continue
            destination = DATA / "raw" / person.name / args.session / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
                    parser.error(f"Já existe outra imagem com esse nome: {destination}")
                skipped += 1
                continue
            shutil.copy2(source, destination)
            copied += 1
    print(f"{copied} imagens copiadas; {skipped} já presentes. Originais preservados.")


if __name__ == "__main__":
    main()
