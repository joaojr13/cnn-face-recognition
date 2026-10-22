"""Reconhecimento sem interface: python main.py foto.jpg --output output/exemplo."""
import argparse
from pathlib import Path
from rostos.pipeline import analyze
from interface.relatorio import annotated_image, png_bytes, report_json, report_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/resultado'))
    args = parser.parse_args()
    try:
        result = analyze(args.image.read_bytes(), args.image.name)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Não foi possível analisar: {exc}\n')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'resultado.json').write_text(report_json(result))
    (args.output/'resultado.txt').write_text(report_text(result))
    (args.output/'resultado.png').write_bytes(png_bytes(annotated_image(result)))
    print(report_text(result))
    print(f'Arquivos salvos em: {args.output.resolve()}')


if __name__ == '__main__':
    main()
