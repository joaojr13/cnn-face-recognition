"""Oito etapas navegáveis, ilustradas com dados da execução selecionada."""
import streamlit as st
from . import textos
from .nomes import display_name
from .relatorio import preview_image, annotated_image


def show_steps(result):
    st.subheader('Como chegamos a este resultado?')
    with st.expander('Treinamento e reconhecimento: o que muda em cada momento'):
        st.markdown(textos.TREINAMENTO)
    if not result['faces']:
        st.info('A execução terminou na detecção: nenhum rosto encontrado. Tente uma foto mais próxima, de frente e com boa luz.')
        return
    selected = st.selectbox('Rosto para acompanhar', range(len(result['faces'])),
        format_func=lambda i: f"{i+1} · {display_name(result['faces'][i]['label'])}", key='trace_face')
    face = result['faces'][selected]
    for index, (title, text) in enumerate(textos.PASSOS, 1):
        with st.expander(title, expanded=index == 8):
            st.markdown(text)
            if index == 1:
                st.image(preview_image(result['image']), width=600)
                st.caption(f"Após leitura: {result['width']} × {result['height']} pixels. Fonte: {result['report']['source']['filename']}.")
            elif index == 2:
                st.image(annotated_image(result), width=600)
                st.write(f"Caixa do rosto {face['id']}: {face['box']}. Pontuação da detecção: {face['detection_score']:.2%}.")
                st.caption('As quatro coordenadas indicam x inicial, y inicial, x final e y final do retângulo na foto preparada. '
                           'Os nomes desenhados vêm do resultado final; a etapa de detecção calcula as caixas.')
            elif index == 3:
                st.image(preview_image(face['crop']), width=240, caption='Recorte com a margem de 8%')
            elif index == 4:
                st.image(preview_image(face['processed']), width=240, caption='Entrada padronizada: 160 × 160 × 3')
                st.caption(f"Código calculado a partir dos pixels da entrada (SHA256), usado para conferir se ela mudou: {face['input_sha256']}")
            elif index == 5:
                example = face['normalization_example']
                st.code(f"Pixel (x=80, y=80), RGB: {example['rgb']}\nValores normalizados: {[round(x, 5) for x in example['normalized_rgb']]}")
                st.caption(f"Regra efetiva: {face['normalization']}. Faixa nesta imagem: {example['min']:.4f} a {example['max']:.4f}.")
            elif index == 6:
                if face.get('activation_maps'):
                    cols = st.columns(4)
                    for i, item in enumerate(face['activation_maps']):
                        with cols[i % 4]:
                            st.image(preview_image(item['image']), caption=f"Canal {item['channel']}", width='stretch')
                            st.caption(f"Mín. {item['min']:.3f} · máx. {item['max']:.3f}")
                    st.caption(f"Saída da base: {face['feature_count']} características.")
                else:
                    st.info('Esta execução não disponibiliza mapas de ativação.')
            elif index == 7:
                ranking = sorted(face.get('scores_by_class', {}).items(), key=lambda x: x[1], reverse=True)
                for name, score in ranking:
                    st.progress(float(score), text=f'{display_name(name)}: {score:.2%}')
                if not ranking:
                    st.info('Ainda não há um modelo treinado.')
            else:
                if face['score'] is None:
                    st.info('Só detecção: modelo não disponível.')
                else:
                    top = max(face['scores_by_class'], key=face['scores_by_class'].get)
                    sign = '≥' if face['accepted'] else '<'
                    message = f"{display_name(top)}: {face['score']:.2%} {sign} {face['threshold']:.2%} → {display_name(face['label'])}"
                    (st.success if face['accepted'] else st.warning)(message)
                    st.caption(f"Distância ao limiar: {100 * (face['score'] - face['threshold']):+.2f} pontos percentuais.")
                    if (result.get('model') or {}).get('threshold_calibrated'):
                        st.caption('Este limiar foi escolhido na validação, equilibrando integrantes recusados e '
                                   'pessoas externas aceitas. A explicação completa está na página Avaliação.')
                    else:
                        st.caption('Limiar provisório: ainda não foi calibrado para este modelo.')
