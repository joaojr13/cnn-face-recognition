"""Monta uma amostra LFW para validação/teste de rejeição; não treina nem classifica.

Origem e SHA256 do arquivo: sklearn/datasets/_lfw.py (LFW original, sem funneling).
A seleção usa ordem pseudoaleatória fixa, duas fotos por identidade e apenas um
critério técnico: exatamente um rosto detectado pelo pré-processamento do projeto.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import shutil
import sys
import tarfile
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rostos.config import ROOT, DATA
from rostos.dados import images_in, unknown_records
from rostos.imagem import read_image, detect

URL = 'https://ndownloader.figshare.com/files/5976018'
SHA256 = '055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0'
SOURCE = 'https://figshare.com/articles/dataset/lfw_tgz/3829986'
REFERENCE = 'https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fetch_archive(cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / 'lfw.tgz'
    if path.exists() and digest(path) == SHA256:
        return path
    partial = path.with_suffix('.part')
    print('Baixando o arquivo LFW original (~172 MB); apenas a amostra será adicionada à base.', flush=True)
    request = urllib.request.Request(URL, headers={'User-Agent': 'AP02-academic-dataset-preparation/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if digest(partial) != SHA256:
        partial.unlink()
        raise ValueError('SHA256 do arquivo recebido diverge da referência scikit-learn.')
    partial.replace(path)
    return path


def candidate_groups(members, seed, per_person):
    groups = defaultdict(list)
    for member in members:
        parts = PurePosixPath(member.name).parts
        if (member.isfile() and len(parts) == 3 and parts[0] == 'lfw'
                and '..' not in parts and parts[2].lower().endswith('.jpg')
                and 0 < member.size <= 12 * 1024 * 1024):
            groups[parts[1]].append(member)
    rng = random.Random(seed)
    identities = sorted(name for name, rows in groups.items() if len(rows) >= per_person)
    rng.shuffle(identities)
    result = []
    for identity in identities:
        photos = sorted(groups[identity], key=lambda row: row.name)
        rng.shuffle(photos)
        result.append((identity, photos))
    return result


def verify(destination=DATA/'unknown'):
    manifest = json.loads((destination/'manifest.json').read_text())
    rows = unknown_records(destination)
    records = manifest['records']
    paths = {str(Path(row['path']).relative_to(destination)): row for row in rows}
    if set(paths) != {row['relative_path'] for row in records}:
        raise ValueError('A pasta e o manifesto têm arquivos diferentes.')
    identities = {'validation': set(), 'test': set()}
    for row in records:
        file = (destination/row['relative_path']).resolve()
        if not file.is_relative_to(destination.resolve()):
            raise ValueError('Caminho fora da base.')
        if digest(file) != row['sha256']:
            raise ValueError(f'Arquivo alterado: {row["relative_path"]}')
        if paths[row['relative_path']]['sha256'] != row['sha256']:
            raise ValueError('Hash inconsistente.')
        identities[row['split']].add(row['source_identity'])
    if identities['validation'] & identities['test']:
        raise ValueError('Identidade repetida entre validação e teste.')
    print(f"Base íntegra: {len(records)} fotos; "
          f"{len(identities['validation'])} pessoas na validação e {len(identities['test'])} no teste.", flush=True)
    return manifest


def build(validation_people=25, test_people=25, per_person=2, seed=42):
    destination = DATA/'unknown'
    if (destination/'manifest.json').exists():
        verify(destination)
        print('Base já existente preservada. Não houve nova seleção.', flush=True)
        return
    existing = [p for p in destination.iterdir() if p.name not in {'.gitkeep', '.DS_Store'}]
    if existing:
        raise ValueError('A pasta unknown já contém arquivos. Não serão substituídos.')
    archive = fetch_archive(ROOT/'.cache/datasets')
    known_hashes = {digest(p) for p in images_in(DATA/'raw')}
    needed = validation_people + test_people
    rejected, selected, seen = [], [], set(known_hashes)
    # extractfile lê bytes; nunca extrai caminhos ou links do TAR no filesystem.
    with tempfile.TemporaryDirectory(prefix='.unknown-lfw-', dir=DATA) as temporary:
        pending = Path(temporary)
        with tarfile.open(archive, 'r:gz') as tar:
            for identity, members in candidate_groups(tar.getmembers(), seed, per_person):
                accepted = []
                for member in members:
                    content = tar.extractfile(member).read()
                    sha = hashlib.sha256(content).hexdigest()
                    reason = None
                    if sha in seen:
                        reason = 'duplicate_content'
                    else:
                        try:
                            image = read_image(content)
                            faces = detect(image)
                            if len(faces) != 1:
                                reason = f'face_count_{len(faces)}'
                        except ValueError as exc:
                            reason = str(exc)
                    if reason:
                        rejected.append({'source_image': member.name, 'reason': reason})
                        continue
                    accepted.append((member, content, sha, image.size, faces[0]))
                    seen.add(sha)
                    if len(accepted) == per_person:
                        break
                if len(accepted) < per_person:
                    rejected.append({'source_identity': identity, 'reason': 'insufficient_usable_images'})
                    continue
                selected.append((identity, accepted))
                print(f'Identidade {len(selected)}/{needed}: {len(accepted)} imagens utilizáveis.', flush=True)
                if len(selected) == needed:
                    break
        if len(selected) != needed:
            raise ValueError('Não foi possível montar a amostra solicitada. A base anterior foi preservada.')
        # Pessoas, não fotografias, são a unidade de divisão.
        random.Random(seed + 1).shuffle(selected)
        records = []
        for index, (identity, photos) in enumerate(selected):
            split = 'validation' if index < validation_people else 'test'
            alias = 'lfw_' + hashlib.sha256(identity.encode()).hexdigest()[:12]
            folder = pending/split/alias
            folder.mkdir(parents=True)
            for number, (member, content, sha, size, face) in enumerate(photos, 1):
                relative = f'{split}/{alias}/foto_{number:02d}.jpg'
                (pending/relative).write_bytes(content)
                records.append({'relative_path':relative, 'path':f'data/unknown/{relative}',
                    'person':'Desconhecido', 'external_identity':alias, 'split':split,
                    'source_identity':identity, 'source_image':member.name,
                    'sha256':sha, 'image_size':list(size), 'box':face['box'],
                    'detection_score':face['detection_score']})
        manifest = {'schema_version':1, 'dataset':'LFW original subset',
            'created_at':datetime.now(timezone.utc).isoformat(), 'seed':seed,
            'source_page':SOURCE, 'archive_url':URL, 'archive_sha256':SHA256,
            'archive_reference':REFERENCE,
            'citation':'Gary B. Huang, Manu Ramesh, Tamara Berg, Erik Learned-Miller. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. UMass Amherst Technical Report 07-49, 2007.',
            'usage_note':'Subconjunto para experimento acadêmico local. O espelho Figshare indica CC BY 4.0; isso não é verificação individual de direitos autorais das fotografias originais.',
            'selection':'Identidades com pelo menos duas imagens, embaralhamento reprodutível; exatamente um rosto MTCNN >=0.95; sem consultar a CNN de identidade.',
            'limitations':['Amostra condicionada à detecção de um rosto; não mede falhas do detector no conjunto LFW completo.',
                'Não representa a distribuição de fotos do grupo ou da webcam.',
                'Desconhecido significa externo ao grupo, não necessariamente ausente do pré-treinamento VGGFace2.',
                'A separação é por identidade LFW; não há comprovação de sessões fotográficas independentes dentro da identidade.',
                'Teste não executado; não escolher ajustes com suas previsões.'],
            'validation_people':validation_people, 'test_people':test_people,
            'images_per_person':per_person, 'recognition_executed':False,
            'records':records, 'excluded_during_preparation':rejected}
        (pending/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        verify(pending)
        # Destino contém somente marcadores; publica cada divisão e o manifesto por último.
        for name in ['validation','test','manifest.json']:
            (pending/name).replace(destination/name)
    verify(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--validation-people', type=int, default=25)
    parser.add_argument('--test-people', type=int, default=25)
    parser.add_argument('--images-per-person', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.validation_people,args.test_people,args.images_per_person) < 1:
        parser.error('Quantidades devem ser positivas.')
    if args.verify:
        verify()
    else:
        build(args.validation_people,args.test_people,args.images_per_person,args.seed)


if __name__ == '__main__':
    main()
