import hashlib
import io
import json
from pathlib import Path
import pytest
from PIL import Image
from rostos.desconhecidos import unknown_summary, validation_preview


def make_base(root):
    records=[]
    for split, color, identity in [('validation','red','person_a'),('test','blue','person_b')]:
        path = root/split/identity/'foto.jpg'
        path.parent.mkdir(parents=True)
        buffer=io.BytesIO()
        Image.new('RGB',(90,100),color).save(buffer,'JPEG')
        path.write_bytes(buffer.getvalue())
        records.append({'relative_path':str(path.relative_to(root)), 'external_identity':identity,
                        'source_identity':identity, 'split':split,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'box':[10,10,70,80]})
    manifest={'records':records,'source_page':'https://example.org/dataset','dataset':'fixture','seed':42}
    (root/'manifest.json').write_text(json.dumps(manifest))
    return records,manifest


def test_split_inventory_and_training_separation(tmp_path):
    records,_=make_base(tmp_path)
    result=unknown_summary(tmp_path)
    assert result['splits']==[{'split':'validation','people':1,'images':1},{'split':'test','people':1,'images':1}]
    assert result['validation_images']==records[:1]
    assert validation_preview(records[0],tmp_path)[2].size==(160,160)
    with pytest.raises(ValueError,match='somente'):
        validation_preview(records[1],tmp_path)


def test_hash_changes_and_unknown_base_missing(tmp_path):
    assert unknown_summary(tmp_path)=={'available':False}
    records,_=make_base(tmp_path)
    (tmp_path/records[0]['relative_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='mudou'):
        unknown_summary(tmp_path)
    with pytest.raises(ValueError,match='mudou'):
        validation_preview(records[0],tmp_path)


def test_source_identity_overlap_is_blocked_even_with_distinct_folder_aliases(tmp_path):
    records,manifest=make_base(tmp_path)
    records[1]['source_identity']=records[0]['source_identity']
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='mesma identidade'):
        unknown_summary(tmp_path)


def test_unlisted_file_and_path_escape_are_blocked(tmp_path):
    records,_=make_base(tmp_path)
    outside=tmp_path.parent/'outside.jpg'
    outside.write_bytes(b'example')
    with pytest.raises(ValueError,match='fora'):
        validation_preview({**records[0],'relative_path':'../outside.jpg'},tmp_path)
    (tmp_path/'validation/person_a/extra.jpg').write_bytes(b'extra')
    with pytest.raises(ValueError,match='difere'):
        unknown_summary(tmp_path)
