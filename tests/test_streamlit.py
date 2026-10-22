import io
from pathlib import Path
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from interface.sessao import input_key, sync_input
from interface.nomes import display_report
from interface.relatorio import report_json
from rostos import pipeline, estudo
from rostos.config import ROOT


def test_input_or_model_change_clears_result_without_cross_session_leak():
    first, second = {}, {}
    key = input_key(b'photo', 'a.jpg', {'run_id': 'a'}, 'upload')
    sync_input(first, key)
    first['analysis_result'] = {'faces': []}
    sync_input(first, key)
    assert 'analysis_result' in first and not second
    for changed in [input_key(b'other', 'a.jpg', {'run_id':'a'}, 'upload'),
                    input_key(b'photo', 'a.jpg', {'run_id':'b'}, 'upload')]:
        sync_input(first, changed)
        assert 'analysis_result' not in first
        first['analysis_result'] = {}
    sync_input(first, None)
    assert 'analysis_result' not in first


def test_display_names_does_not_mutate_model_labels():
    data = {'labels': ['joao', 'samanta'], 'faces':[{'label':'fernanda', 'scores_by_class':{'diego':.8}}]}
    shown = display_report(data)
    assert shown['labels'] == ['Joao', 'Samanta']
    assert shown['faces'][0]['scores_by_class'] == {'Diego': .8}
    assert data['labels'][0] == 'joao'


def test_manifest_relative_paths_and_gallery_are_portable():
    from rostos.modelo import active_metadata
    metadata = active_metadata()
    records = estudo.training_records(metadata)
    assert len(records) == metadata['training_images']
    assert len(records) > 0
    assert all(not Path(row['path']).is_absolute() for row in records)
    assert estudo.training_image(metadata, 0)['processed']
    assert estudo.study_summary(metadata)['training_diagnostic']['size'] == len(records)


def mock_detection(monkeypatch):
    monkeypatch.setattr(pipeline, 'detect', lambda image: [{'box':[4,4,40,40], 'detection_score':.99}])
    monkeypatch.setattr(pipeline.recognizer, 'predict', lambda batch, metadata: [
        {'label':'Desconhecido', 'score':.895, 'accepted':False, 'threshold':.9,
         'scores_by_class':{'samanta':.895,'fernanda':.105}, 'activation_maps':[], 'feature_count':512} for x in batch])


def test_streamlit_pages_and_rejection_trace(monkeypatch):
    mock_detection(monkeypatch)
    at = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
    assert not at.exception
    assert at.button[0].disabled
    at.radio[0].set_value('Exemplo da base').run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.get('download_button')) == 3
    assert any('89.50%' in warning.value and 'Desconhecido' in warning.value for warning in at.warning)
    for page in ['Avaliação', 'Base de treinamento', 'Entenda o projeto']:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, str(at.exception)
    at.sidebar.radio[0].set_value('Reconhecer').run()
    at.radio[0].set_value('Tirar foto agora').run()
    assert not at.exception
    assert len(at.get('camera_input')) == 1
    assert at.button[0].disabled
