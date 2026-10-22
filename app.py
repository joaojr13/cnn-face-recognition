"""Interface Streamlit; o fluxo de reconhecimento está em rostos/pipeline.py."""
import streamlit as st
from rostos.modelo import active_metadata
from interface.componentes import recognize_page, dataset_page, evaluation_page, study_page

st.set_page_config(page_title='Rostos · Laboratório CNN', page_icon='🔎', layout='wide')
st.title('Rostos · Laboratório CNN')
st.caption('AP02 · Processamento de imagens · Reconhecimento facial com redes convolucionais')
try:
    metadata = active_metadata()
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()
with st.sidebar:
    st.header('Explore o projeto')
    page = st.radio('Seção', ['Reconhecer', 'Base de treinamento', 'Avaliação', 'Entenda o projeto'])
    st.divider()
    if metadata:
        st.caption('Modelo ativo')
        st.write(metadata['architecture'])
        st.caption(f"{len(metadata['labels'])} integrantes · {metadata.get('training_images', '?')} fotos")
        st.caption(f"Limiar: {metadata['threshold']:.0%}")
        if metadata.get('experimental'):
            st.warning('Experimento sem teste independente')
    else:
        st.info('Somente detecção: modelo ainda não treinado.')
    st.caption('Uploads ficam na sessão e não entram automaticamente no treinamento.')
{'Reconhecer': recognize_page, 'Base de treinamento': dataset_page,
 'Avaliação': evaluation_page, 'Entenda o projeto': study_page}[page](metadata)
st.divider()
st.caption('As fotografias são processadas no servidor que executa esta aplicação. O resultado é uma previsão experimental.')
