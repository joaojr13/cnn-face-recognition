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
    if result['mode'] == 'experimental' and (metadata or {}).get('threshold_calibrated'):
        st.info('Modelo treinado com duas fotos por integrante. O limiar foi calibrado e há um teste pequeno: '
                'veja a página Avaliação.')
    elif result['mode'] == 'experimental':
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


def show_figure(fig, filename, label):
    import io
    import matplotlib.pyplot as plt
    st.pyplot(fig)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=160)
    st.download_button(label, buffer.getvalue(), filename, 'image/png', key=filename)
    plt.close(fig)


def show_matrix(labels, matrix, title, filename='matriz-confusao.png'):
    from rostos.graficos import confusion_figure
    show_figure(confusion_figure(labels, matrix, title), filename, 'Baixar matriz PNG')


def pct(value):
    return '—' if value is None else f'{value:.1%}'


def class_table(block, labels):
    report = block['classification_report']
    rows = [{'Classe': display_name(name), 'Fotos': int(report[name]['support']),
             'Precisão': f"{report[name]['precision']:.3f}", 'Recall': f"{report[name]['recall']:.3f}",
             'F1': f"{report[name]['f1-score']:.3f}"}
            for name in labels if report[name]['support'] > 0]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


def sample_table(block, only_members=False):
    rows = [{'Foto': s['path'].split('/')[-2] + '/' + s['path'].split('/')[-1],
             'Correto': display_name(s['truth']), 'Resposta': display_name(s['prediction']),
             'Maior candidato': display_name(s['top_candidate']), 'Pontuação': f"{s['score']:.2%}",
             'Acerto': '✓' if s['truth'] == s['prediction'] else '✗'}
            for s in block['samples'] if not (only_members and s['truth'] == 'Desconhecido')]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


def fmt(value, digits=3):
    return f'{value:.{digits}f}'


def threshold_explanation(metrics):
    """Por que o limiar calibrado é o ponto de equilíbrio, com os dados desta avaliação."""
    import numpy as np
    from rostos.avaliacao_facial import score_panels, threshold_candidates, threshold_effect
    from rostos.graficos import score_figure, threshold_curve_figure
    labels = metrics['labels'][:-1]
    threshold, provisional = metrics['threshold'], metrics['provisional_threshold']
    calibration = metrics['calibration']
    validation, test = metrics['validation']['samples'], metrics['test']['samples']
    members = [s for s in validation if s['truth'] != 'Desconhecido']
    outsiders = [s for s in validation if s['truth'] == 'Desconhecido']

    st.subheader(f'Como chegamos ao limiar {fmt(threshold)}')
    st.markdown(textos.EQUILIBRIO)

    st.markdown('**1. Onde ficam as pontuações.** Cada ponto é uma foto, posicionada pela maior pontuação do softmax. '
                'Integrantes à direita do limiar recebem um nome; externos à direita são aceitos por engano.')
    show_figure(score_figure(score_panels(metrics), threshold, provisional),
                'distribuicao-pontuacoes.png', 'Baixar gráfico PNG')
    lowest = min(members, key=lambda s: s['score'])
    highest = max(outsiders, key=lambda s: s['score'])
    st.write(f"Na validação, o integrante com menor pontuação foi **{display_name(lowest['truth'])}** "
             f"({fmt(lowest['score'])}), e o externo com maior pontuação recebeu **{fmt(highest['score'])}** "
             f"para **{display_name(highest['top_candidate'])}**. Como o externo ficou acima do integrante, "
             'nenhum limiar separa as duas listas sem erro. É preciso escolher qual erro aceitar.')

    st.markdown('**2. O efeito de cada limiar.** As linhas mostram, para cada limiar possível, quantas fotos '
                'da validação seriam erradas de cada tipo.')
    grid = np.unique(np.concatenate([np.linspace(0.70, 1.0, 601), [s['score'] for s in validation]]))
    rows = [threshold_effect(validation, labels, value) for value in grid if value >= 0.70]
    show_figure(threshold_curve_figure(rows, threshold, provisional, calibration['target_max_far']),
                'efeito-do-limiar.png', 'Baixar gráfico PNG')

    st.markdown('**3. Comparação de limiares.** A escolha usa só as colunas da validação. '
                'As colunas do teste mostram, depois, o que aconteceu em fotos que não participaram da escolha.')
    table = []
    for value, note in threshold_candidates(metrics):
        val, tst = threshold_effect(validation, labels, value), threshold_effect(test, labels, value)
        table.append({'Limiar': fmt(value), 'Observação': note,
                      'Validação: integrantes aceitos': f"{val['members_correct']}/{val['members']}",
                      'Validação: externos aceitos': f"{val['outsiders_accepted']}/{val['outsiders']}",
                      'Validação: F1 macro': fmt(val['macro_f1']),
                      'Teste: integrantes aceitos': f"{tst['members_correct']}/{tst['members']}",
                      'Teste: externos aceitos': f"{tst['outsiders_accepted']}/{tst['outsiders']}"})
    st.dataframe(pd.DataFrame(table), hide_index=True, width='stretch')
    accept_all = threshold_effect(validation, labels, lowest['score'])
    chosen = threshold_effect(validation, labels, threshold)
    if accept_all['outsiders_accepted'] > chosen['outsiders_accepted']:
        st.write(f"Para aceitar todos os integrantes da validação, o limiar teria de descer para "
                 f"{fmt(lowest['score'])}. Isso deixaria passar {accept_all['outsiders_accepted']} externos "
                 f"(F1 macro {fmt(accept_all['macro_f1'])}). O limiar {fmt(threshold)} recusa "
                 f"{chosen['members_rejected']} foto(s) de integrante e aceita {chosen['outsiders_accepted']} externo(s) "
                 f"(F1 macro {fmt(chosen['macro_f1'])}). Pela regra, esse é o equilíbrio: limiares mais altos não "
                 'melhoram o F1 da validação e, no empate, fica o menor, que recusa menos integrantes.')
    old, new = metrics['test_at_provisional_threshold'], metrics['test']
    st.write(f"**Confirmação no teste:** com {fmt(threshold)}, {pct(new['unknown_false_accept_rate'])} dos externos "
             f"foram aceitos e {pct(new['known_rejection_rate'])} dos integrantes foram recusados. "
             f"Com o provisório {fmt(provisional, 2)}, seriam {pct(old['unknown_false_accept_rate'])} e "
             f"{pct(old['known_rejection_rate'])}.")

    st.markdown('**4. Experimente outro limiar.** A simulação reaplica o limiar às pontuações já salvas. '
                'O modelo ativo continua usando o limiar calibrado.')
    value = st.slider('Limiar simulado', 0.50, 1.00, float(round(threshold, 3)), 0.001, format='%.3f',
                      key='limiar_simulado')
    for column, title, samples in zip(st.columns(2), ['Validação', 'Teste'], [validation, test]):
        effect = threshold_effect(samples, labels, value)
        with column:
            st.markdown(f'**{title}**')
            st.metric('Integrantes aceitos com o nome certo', f"{effect['members_correct']}/{effect['members']}")
            st.metric('Externos aceitos por engano', f"{effect['outsiders_accepted']}/{effect['outsiders']}")
            st.metric('F1 macro', fmt(effect['macro_f1']))


def facial_evaluation(metrics):
    """Resultado de rostos.avaliacao_facial: teste, validação cruzada e limiar."""
    test, cross = metrics['test'], metrics['cross_validation']
    labels = metrics['labels']
    st.markdown(textos.PROTOCOLO)
    st.caption(f"Limiar calibrado: {metrics['threshold']:.4f} · provisório anterior: "
               f"{metrics['provisional_threshold']:.2f} · avaliação de {metrics['created_at'][:10]}")

    st.subheader('Teste independente')
    st.caption(f"{test['known_images']} fotos novas de integrantes + {test['unknown_images']} fotos de pessoas externas. "
               'Nenhuma participou do treino nem da escolha do limiar.')
    for col, title, value in zip(st.columns(4), ['Acurácia', 'Precisão macro', 'Recall macro', 'F1 macro'],
                                 [test['accuracy'], test['macro_precision'], test['macro_recall'], test['macro_f1']]):
        col.metric(title, pct(value))
    for col, title, value in zip(st.columns(4),
            ['Integrantes rejeitados', 'Externos aceitos', 'Acerto só entre integrantes', 'Limiar'],
            [pct(test['known_rejection_rate']), pct(test['unknown_false_accept_rate']),
             pct(test['known_closed_set_accuracy']), f"{metrics['threshold']:.3f}"]):
        col.metric(title, value)
    show_matrix(labels, test['confusion_matrix'], 'Matriz de confusão · teste independente',
                'matriz-confusao-teste.png')
    class_table(test, labels)
    with st.expander('Resposta para cada foto de integrante no teste'):
        sample_table(test, only_members=True)
        st.caption('"Maior candidato" é o nome com maior pontuação antes do limiar; '
                   'quando a pontuação fica abaixo do limiar, a resposta é Desconhecido.')
    old = metrics['test_at_provisional_threshold']
    with st.expander(f"Comparação: o mesmo teste com o limiar provisório de {metrics['provisional_threshold']:.0%}"):
        cols = st.columns(3)
        cols[0].metric('Acurácia', pct(old['accuracy']))
        cols[1].metric('Integrantes rejeitados', pct(old['known_rejection_rate']))
        cols[2].metric('Externos aceitos', pct(old['unknown_false_accept_rate']))
        show_matrix(labels, old['confusion_matrix'], 'Teste com limiar provisório', 'matriz-confusao-provisorio.png')

    threshold_explanation(metrics)

    st.subheader('Validação cruzada (complementar)')
    st.caption(f"Leave-one-out com {cross['known_images']} fotos dos integrantes (treino + teste): cada foto é "
               'prevista por uma camada final treinada sem ela, com o limiar já escolhido. '
               'Dá mais exemplos por pessoa, mas não avalia o modelo ativo nem a rejeição de externos.')
    for col, title, value in zip(st.columns(4), ['Acurácia', 'Precisão macro', 'Recall macro', 'F1 macro'],
                                 [cross['accuracy'], cross['macro_precision'], cross['macro_recall'], cross['macro_f1']]):
        col.metric(title, pct(value))
    show_matrix(labels, cross['confusion_matrix'], 'Matriz de confusão · validação cruzada (integrantes)',
                'matriz-confusao-validacao-cruzada.png')
    class_table(cross, labels)
    with st.expander('Resposta para cada foto na validação cruzada'):
        sample_table(cross)
    with st.expander('Limitações desta avaliação'):
        for item in metrics.get('limitations', []):
            st.write(f'- {item}')
    st.download_button('Avaliação completa JSON', json.dumps(display_report(metrics), ensure_ascii=False, indent=2),
                       'avaliacao.json', 'application/json')


def evaluation_page(metadata):
    st.header('Avaliação: o que os números demonstram?')
    st.markdown(textos.AVALIACAO)
    if not metadata:
        st.info('Não há modelo ativo.')
        return
    independent = evaluation(metadata)
    if independent['available'] and 'test' in independent['metrics']:
        facial_evaluation(independent['metrics'])
    elif independent['available']:
        metrics = independent['metrics']
        st.subheader('Teste independente')
        show_matrix(metrics['labels'], metrics['confusion_matrix'], 'Matriz de confusão · teste independente')
        st.json(display_report(metrics))
    else:
        st.subheader('Teste independente')
        st.info('Pendente para o modelo ativo. Execute no terminal: '
                '`.venv/bin/python -m rostos.avaliacao_facial --aplicar-limiar`')
    st.divider()
    study = study_summary(metadata)
    diagnostic = study['training_diagnostic']
    st.subheader('Diagnóstico nas fotos de treinamento')
    if diagnostic:
        st.warning(f"Estas {diagnostic['size']} fotos já foram usadas para treinar o modelo. Os números não estimam a taxa de acerto em fotos novas.")
        for col, title, key in zip(st.columns(4), ['Acurácia', 'Precisão macro', 'Recall macro', 'F1 macro'],
                                   ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']):
            value = diagnostic[key]
            col.metric(title, '—' if value is None else f'{value:.1%}')
        show_matrix(diagnostic['labels'], diagnostic['matrix'], 'Matriz de confusão · fotos de treino',
                    'matriz-confusao-treino.png')
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
