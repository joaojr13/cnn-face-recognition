"""Fotos/resultados pertencem à sessão; somente o motor é compartilhado."""
import hashlib
import json
import streamlit as st
from rostos import pipeline

PIPELINE_VERSION = 'streamlit-crop8-pad160-v1'

@st.cache_resource(show_spinner=False)
def motor():
    # O módulo tem RLock: detecção, carregamento e hooks nunca concorrem.
    return pipeline


def input_key(content, filename, metadata, source):
    return hashlib.sha256(content + json.dumps([filename, metadata, source, PIPELINE_VERSION],
                                               sort_keys=True).encode()).hexdigest()


def sync_input(state, key):
    if state.get('analysis_key') != key:
        state.pop('analysis_result', None)
        state['analysis_key'] = key
