"""Confere se os pesos e as fotos migradas mantêm os hashes registrados."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rostos.config import ROOT, ARTIFACTS
from rostos.modelo import active_metadata, artifact_path
from rostos.estudo import training_records
from rostos.arquivos import resolve_record_path


def main():
    record = json.loads((ROOT/'docs/MIGRACAO.json').read_text())
    path = ARTIFACTS / record['active_run'] / 'model.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['model_sha256'], 'Pesos migrados alterados'
    for item in record['files']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    active = active_metadata()
    artifact_path(active)
    for item in training_records(active):
        path = resolve_record_path(item['path'])
        assert path.is_relative_to((ROOT/'data/raw').resolve()), 'Imagem fora da base'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['path']
    print(f"Pacote íntegro: {sum(Path(item['path']).suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'} for item in record['files'])} fotos migradas e pesos preservados. Modelo ativo: {active['run_id']}")


if __name__ == '__main__':
    main()
