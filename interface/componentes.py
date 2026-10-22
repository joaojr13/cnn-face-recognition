"""Páginas Streamlit: apresentação e interação, sem lógica de reconhecimento."""
import json
import logging
import pandas as pd
import streamlit as st
from rostos.config import ROOT
from rostos.dados import inventory, images_in
from rostos.estudo import study_summary, training_image, evaluation
from rostos.pipeline import processing_lock
from . import textos
from .desconhecidos import unknown_dataset_panel
from .nomes import display_name, display_report
from .relatorio import annotated_image, png_bytes, preview_image, report_json, report_text
from .sessao import input_key, sync_input, motor
from .passos import show_steps


def recognize_page(metadata):
    st.header('Reconhecer integrantes')
    st.markdown(textos.INTRO)
    source = st.radio('Origem da fotografia', ['Enviar arquivo', 'Tirar foto agora', 'Exemplo da base'], horizontal=True)
    content, filename = None, ''
    if source == 'Enviar arquivo':
        upload = st.file_uploader('Selecione uma foto individual ou do grupo', type=['jpg', 'jpeg', 'png', 'webp'])
        if upload is not None:
            content, filename = upload.getvalue(), upload.name
    elif source == 'Tirar foto agora':
        st.caption('Autorize a câmera, enquadre o rosto e capture uma fotografia. Somente a foto é enviada para análise.')
        photo = st.camera_input('Capturar fotografia')
        if photo is not None:
            content, filename = photo.getvalue(), 'webcam.jpg'
    else:
        samples = images_in(ROOT / 'data/raw')
        if samples:
            sample = st.selectbox('Escolha uma foto já usada no treinamento', samples,
                format_func=lambda p: f"{display_name(p.relative_to(ROOT / 'data/raw').parts[0])} · {p.name}")
            content, filename = sample.read_bytes(), sample.name
            st.info('Este exemplo faz parte da base. Reconhecê-lo não demonstra desempenho em uma foto nova.')
        else:
            st.info('Não há exemplos disponíveis.')
    key = input_key(content, filename, metadata, source) if content is not None else None
    sync_input(st.session_state, key)
    if st.button('Identificar rostos', type='primary', disabled=content is None):
        st.session_state.pop('analysis_result', None)
        try:
            with st.spinner('Localizando rostos e executando a CNN…'):
                st.session_state['analysis_result'] = motor().analyze(content, filename)
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            logging.exception('Falha ao reconhecer imagem')
            st.error('Não foi possível executar o modelo. Confira as dependências e os arquivos do modelo no servidor.')
    result = st.session_state.get('analysis_result')
    if not result:
        return
    st.subheader('Resultado')
    st.image(annotated_image(result), width='stretch')
    st.caption(f"{len(result['faces'])} rosto(s) detectado(s).")
    if result['mode'] == 'experimental':
        st.warning('Resultado experimental: modelo treinado com poucas fotos, sem teste independente e com limiar provisório.')
    for face in result['faces']:
        columns = st.columns([1, 5])
        columns[0].image(preview_image(face['processed']), width=100)
        with columns[1]:
            st.write(f"**{face['id']} · {display_name(face['label'])}**")
            if face['score'] is not None:
                st.caption(f"Maior pontuação: {face['score']:.2%} · limiar: {face['threshold']:.2%}")
    with st.expander('Salvar resultado e conferir a execução'):
        st.caption(f"Modelo: {(result['model'] or {}).get('run_id', 'nenhum')} · {result['report']['analyzed_at']}")
        st.code(result['report']['source']['sha256'], language=None)
        cols = st.columns(3)
        cols[0].download_button('Imagem anotada', png_bytes(annotated_image(result)), 'resultado.png', 'image/png')
        cols[1].download_button('Relatório JSON', report_json(result), 'resultado.json', 'application/json')
        cols[2].download_button('Resumo TXT', report_text(result), 'resultado.txt', 'text/plain')
    show_steps(result)


def dataset_page(metadata):
    st.header('Base de treinamento')
    data = inventory()
    st.write('A galeria usa a lista de fotos registrada quando o modelo ativo foi treinado, chamada de manifesto. '
             'Antes de exibir uma imagem, o programa confere seu hash: um código calculado a partir do conteúdo do arquivo. '
             'Se o arquivo tiver mudado, a reprodução é bloqueada para não mostrar outra imagem como se fosse a usada no treino.')
    st.dataframe(pd.DataFrame([{'Integrante': display_name(p['name']), 'Fotos na pasta': p['count'],
                               'Pastas de sessão': len(p['sessions'])} for p in data['people']]), hide_index=True, width='stretch')
    if not metadata:
        st.info('Ainda não há um treinamento salvo com uma lista de fotos para consultar.')
        return
    study = study_summary(metadata)
    records = study['training_images']
    st.caption(f"Modelo ativo: {len(records)} fotos originais. Pasta atual: {data['total_images']} fotos. Adicionar arquivos não retreina a rede automaticamente.")
    st.markdown(textos.TREINAMENTO)
    if records:
        index = st.selectbox('Foto registrada no treinamento', range(len(records)), format_func=lambda i:
            f"{display_name(records[i]['person'])} · {records[i]['filename']}")
        try:
            with processing_lock:
                photo = training_image(metadata, index)
            columns = st.columns(3)
            for col, key, caption in zip(columns, ['original', 'crop', 'processed'], ['Original', 'Recorte', 'Entrada da CNN']):
                col.image(preview_image(photo[key]), caption=caption, width='stretch')
            st.caption(f"Integrante: {display_name(photo['person'])} · pasta de origem: {records[index]['session']} · uso: treinamento")
            with st.expander('Variações da foto e verificação do arquivo'):
                st.code(photo['sha256'], language=None)
                st.caption('Nove entradas derivadas da mesma foto; elas não aumentam o número de observações independentes.')
                cols = st.columns(3)
                for i, variant in enumerate(photo['variants']):
                    cols[i % 3].image(preview_image(variant['image']), caption=variant['name'], width=180)
        except (ValueError, FileNotFoundError, IndexError) as exc:
            st.error(str(exc))
    with st.expander('Como as fotografias influenciam o reconhecimento'):
        st.write('Fotos de frente e com boa luz facilitam a detecção e preservam detalhes do rosto. '
                 'Fotografias de ocasiões diferentes incluem mudanças reais de expressão, óculos e iluminação. '
                 'Essa variedade ajuda a reduzir a dependência de condições específicas do treino. '
                 'As pastas de sessão organizam os arquivos, mas seus nomes não comprovam que as fotos vieram de ocasiões diferentes. '
                 'A separação entre treino, validação e teste mantém as imagens usadas para aprender distintas das usadas para escolher ajustes ou avaliar o resultado.')

    st.divider()
    unknown_dataset_panel()


def show_matrix(labels, matrix, title):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 5))
    values = np.asarray(matrix)
    ax.imshow(values, cmap='Blues', vmin=0, vmax=max(1, int(values.max())))
    ax.set_xticks(range(len(labels)), [display_name(x) for x in labels], rotation=35, ha='right')
    ax.set_yticks(range(len(labels)), [display_name(x) for x in labels])
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, str(values[row, col]), ha='center', va='center',
                    color='white' if values[row, col] > values.max() / 2 else 'black')
    ax.set(xlabel='Identidade prevista', ylabel='Identidade correta', title=title)
    fig.tight_layout()
    st.pyplot(fig)
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=160)
    st.download_button('Baixar matriz PNG', buffer.getvalue(), 'matriz-confusao.png', 'image/png', key=title)
    plt.close(fig)


def evaluation_page(metadata):
    st.header('Avaliação: o que os números demonstram?')
    st.markdown(textos.AVALIACAO)
    if not metadata:
        st.info('Não há modelo ativo.')
        return
    study = study_summary(metadata)
    diagnostic = study['training_diagnostic']
    st.subheader('Diagnóstico nas fotos de treinamento')
    if diagnostic:
        st.warning(f"Estas {diagnostic['size']} fotos já foram usadas para treinar o modelo. Os números não estimam a taxa de acerto em fotos novas.")
        for col, title, key in zip(st.columns(4), ['Acurácia', 'Precisão macro', 'Recall macro', 'F1 macro'],
                                   ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']):
            value = diagnostic[key]
            col.metric(title, '—' if value is None else f'{value:.1%}')
        show_matrix(diagnostic['labels'], diagnostic['matrix'], 'Matriz de confusão · fotos de treino')
        rows = [{'Integrante': display_name(row['label']), 'Fotos': row['support'],
                 **{name: '—' if row[key] is None else f"{row[key]:.3f}" for name, key in
                    [('Precisão', 'precision'), ('Recall', 'recall'), ('F1', 'f1')]}} for row in diagnostic['rows']]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.download_button('Diagnóstico de treino JSON', json.dumps(display_report(diagnostic), ensure_ascii=False, indent=2),
                           'diagnostico-treino.json', 'application/json')
    else:
        st.info('Não há diagnóstico compatível com o manifesto desta execução.')
    if study['history'].get('loss'):
        with st.expander('Como a perda mudou durante o treinamento'):
            losses = study['history']['loss']
            st.line_chart(pd.DataFrame({'Época': range(1, len(losses)+1), 'Perda de treino': losses}).set_index('Época'))
            st.caption('A perda mede o erro usado para ajustar a camada final, comparando sua saída com os rótulos corretos. '
                       'O eixo horizontal conta as épocas de treinamento. Uma perda menor indica melhor ajuste ao objetivo nas fotos de treino; '
                       'não mede a taxa de erro em fotos novas.')
    st.subheader('Teste independente')
    independent = evaluation(metadata)
    if not independent['available']:
        st.info('Pendente: ainda não há teste independente nem medição de falsos aceites de pessoas externas para o modelo ativo.')
    else:
        metrics = independent['metrics']
        show_matrix(metrics['labels'], metrics['confusion_matrix'], 'Matriz de confusão · teste independente')
        st.json(display_report(metrics))
    with st.expander('Como ler a matriz e calcular cada métrica', expanded=True):
        st.markdown(textos.METRICAS)


def study_page(metadata):
    st.header('Entenda o projeto')
    st.markdown(textos.VISAO_GERAL)
    with st.expander('Como o modelo aprende e reconhece', expanded=True):
        st.markdown(textos.TREINAMENTO)
        if metadata:
            st.write(f"Modelo ativo: {metadata['architecture']}. Fotos: {metadata.get('training_images')}. "
                     f"Épocas: {metadata.get('epochs_completed')}. Parâmetros treinados: {metadata.get('trainable_parameters', 'não registrado')}.")
    for title, filename in [('Como o reconhecimento funciona', 'COMO_FUNCIONA.md'),
                            ('Mapa do código', 'MAPA_DO_CODIGO.md'), ('Validação e limitações', 'VALIDACAO.md')]:
        with st.expander(title):
            path = ROOT / 'docs' / filename
            if path.exists():
                st.markdown(path.read_text())
                st.download_button('Baixar documento', path.read_text(), filename, 'text/markdown', key=filename)
