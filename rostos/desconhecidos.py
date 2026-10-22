"""Consulta à base externa; não executa reconhecimento nem consome o teste."""
import hashlib
import json
from pathlib import Path
from .config import DATA
from .dados import unknown_records
from .imagem import read_image, crop_face, model_image


def unknown_summary(root=None):
    root = root or DATA/'unknown'
    manifest = root/'manifest.json'
    if not manifest.is_file():
        return {'available':False}
    data = json.loads(manifest.read_text())
    rows = unknown_records(root)
    expected = data['records']
    by_path = {str(Path(row['path']).relative_to(root)):row for row in rows}
    if len(expected) != len(rows) or set(by_path) != {row['relative_path'] for row in expected}:
        raise ValueError('A pasta de desconhecidos difere do manifesto. Confira a base antes de usar.')
    identities = {'validation':set(), 'test':set()}
    for record in expected:
        row = by_path[record['relative_path']]
        if row['sha256'] != record['sha256'] or row['split'] != record['split']:
            raise ValueError('Um arquivo de desconhecido mudou desde a preparação.')
        identities[record['split']].add(record['source_identity'])
    if identities['validation'] & identities['test']:
        raise ValueError('A mesma identidade aparece na validação e no teste.')
    return {'available':True, 'source_page':data['source_page'],
        'dataset':data['dataset'], 'seed':data['seed'],
        'splits':[{'split':split, 'people':len(identities[split]),
                   'images':sum(row['split']==split for row in expected)} for split in ['validation','test']],
        'validation_images':[row for row in expected if row['split']=='validation']}


def validation_preview(record, root=None):
    root = (root or DATA/'unknown').resolve()
    if record['split'] != 'validation':
        raise ValueError('A galeria didática mostra somente a divisão de validação.')
    path = (root/record['relative_path']).resolve()
    if not path.is_relative_to(root/'validation'):
        raise ValueError('Caminho fora da base de validação.')
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record['sha256']:
        raise ValueError('A imagem mudou desde a preparação.')
    original = read_image(content)
    cropped = crop_face(original, record['box'])
    return original, cropped, model_image(cropped)
