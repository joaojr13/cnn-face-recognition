"""Inventário da base externa, separado das fotos usadas no treinamento."""
import pandas as pd
import streamlit as st
from rostos.desconhecidos import unknown_summary, validation_preview


def unknown_dataset_panel():
    st.subheader('Pessoas externas: validação e teste')
    try:
        summary = unknown_summary()
    except (ValueError, OSError, KeyError) as exc:
        st.error(f'Não foi possível conferir a base externa: {exc}')
        return
    if not summary['available']:
        st.info('Ainda não há uma base externa preparada.')
        return
    labels = {'validation':'Validação — para ajustar o limiar', 'test':'Teste — reservado para a avaliação final'}
    st.dataframe(pd.DataFrame([{'Divisão':labels[row['split']], 'Pessoas':row['people'], 'Fotos':row['images']}
                              for row in summary['splits']]), hide_index=True, width='stretch')
    st.markdown(f"Origem: [LFW, espelho usado pelo scikit-learn]({summary['source_page']}). "
                'As pessoas de validação e teste são diferentes. O conteúdo de cada arquivo foi conferido por um código de verificação, chamado hash.')
    st.info('Estas fotos não entram no treinamento dos cinco integrantes. Servem para avaliar a rejeição de pessoas externas. '
            'Ainda não executamos o reconhecimento neste conjunto nem alteramos o limiar.')
    st.caption('A preparação exigiu um rosto detectável por foto. Este subconjunto não mede falhas de detecção no LFW completo '
               'nem representa todas as condições da webcam. Uma identidade externa ao grupo pode ter aparecido no pré-treinamento da CNN.')
    with st.expander('Preparação de uma foto da validação externa'):
        records = summary['validation_images']
        if not records:
            return
        index = st.selectbox('Exemplo de validação externa', range(len(records)),
            format_func=lambda i:f"{records[i]['external_identity']} · {records[i]['relative_path'].split('/')[-1]}")
        try:
            images = validation_preview(records[index])
            for col, image, caption in zip(st.columns(3), images, ['Foto LFW', 'Recorte do rosto', 'Entrada 160 × 160']):
                col.image(image, caption=caption, width='stretch')
            st.caption('Só preparação da imagem: nenhuma pontuação de identidade foi calculada.')
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    st.caption('As fotos de teste permanecem reservadas para avaliar o modelo depois de fixar os ajustes com a validação. Isso evita usar as mesmas previsões para escolher o limiar e medir seu resultado.')
