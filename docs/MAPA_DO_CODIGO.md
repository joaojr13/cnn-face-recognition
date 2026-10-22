# Mapa do código

A organização acompanha a AP01: interface separada do processamento, uma entrada web,
uma entrada de terminal e um fluxo principal curto. Não se copiam algoritmos de placas
para rostos: Canny, morfologia e OCR da AP01 não fazem parte desta solução.

## Caminho de uma fotografia pelo programa

`app.py` → `interface/componentes.py` → `interface/sessao.py` → `rostos/pipeline.py`.
O fluxo `analyze` usa um bloqueio para que sessões simultâneas não misturem carregamento
da rede e captura de ativações. `_analyze` lê os metadados ativos uma única vez e segue:

| Etapa | Código | Entrada → saída |
|---|---|---|
| 1. Leitura | `imagem.read_image` | bytes → Pillow RGB, orientação corrigida, maior lado até 1.600 |
| 2. Localização | `imagem.detect`, `deteccao.SafeStagePNet` | RGB → caixas e pontuações de detecção |
| 3. Recorte | `imagem.crop_face` | imagem + caixa → rosto com margem de 8% |
| 4. Padronização | `imagem.model_image` | recorte → RGB 160 × 160, proporção preservada |
| 5. Normalização | `treinamento_facial.face_tensor` | lote RGB → tensor normalizado N × 3 × 160 × 160 |
| 6. CNN | `FaceClassifier.inspect_numpy` | tensor → 512 características e oito mapas reais por rosto |
| 7. Pontuação | `classifier(...).softmax(dim=1)` | características → pontuações entre os integrantes |
| 8. Decisão | `modelo.decisions` | pontuações + limiar → nome aceito ou Desconhecido |
| Registro | `saida.build_report` | execução → relatório com hashes, modelo, caixas e motivo da decisão |

`etapas/preparacao.py` prepara os recortes e descreve os pixels usados, incluindo o exemplo
numérico de normalização. A normalização efetiva da CNN acontece em `face_tensor`.
`interface/passos.py` apresenta as oito etapas; `interface/textos.py` guarda as explicações.

## Dados e dimensões

- `box = [x1, y1, x2, y2]`: coordenadas na imagem depois da leitura, com limite final
  exclusivo no recorte Pillow. Não são coordenadas da imagem original antes da redução.
- Entrada por rosto: `uint8`, altura 160, largura 160, canais RGB 3.
- Lote da rede: `float32`, quantidade de rostos × 3 × 160 × 160, aproximadamente [−1, 1].
- Características: quantidade de rostos × 512.
- Pontuações: quantidade de rostos × quantidade de integrantes; cada linha soma aproximadamente 1.
- Ativações mostradas: oito canais da primeira camada, 79 × 79 cada, depois de BatchNorm/ReLU.
- `score`: maior pontuação de identidade; `detection_score`: pontuação do detector.
- `accepted`: se passou pelo limiar. Não é confirmação de identidade.
- `analysis_result`: resultado exclusivo da sessão do navegador. Os pesos da CNN são compartilhados.

## Treinamento e relatórios

`treinamento_facial.train_specialized` coleta registros, prepara imagens, cria nove variações,
calcula características com a base congelada e treina a camada final. Salva os pesos,
confere uma restauração e só então pode ativar a execução. A perda fica em `history.json`.
`manifest.json` registra fotos, rótulos, sessões, hashes e caixas. Os caminhos salvos são
relativos à raiz do projeto; mover a pasta não deve quebrar a galeria.

`estudo.training_image` limita a leitura à pasta da base e confere o hash antes de reproduzir
recorte e entrada. `estudo.summarize_training` constrói a matriz e as métricas de treino.
Ela não inventa uma taxa para desconhecidos quando não há exemplos externos.
`estudo.evaluation` impede que um experimento exponha um relatório como teste independente.

## Localização das responsabilidades

| Pergunta | Arquivo |
|---|---|
| Onde altero textos sem mexer no modelo? | `interface/textos.py` e `docs/` |
| Onde estão os limites de upload e tamanho? | `rostos/config.py` |
| Onde está o limiar ativo? | `artifacts/active.json` (registrado pelo treinamento) |
| Onde explicamos um rosto individual? | `interface/passos.py` |
| Onde ficam as fotos utilizadas? | `data/raw/`, referenciadas no manifesto da execução |
| Como evitamos resultado antigo após trocar a foto? | `interface/sessao.py` |
| Onde estão os nomes com maiúscula inicial? | `interface/nomes.py` |
| Como gerar arquivos sem navegador? | `main.py`, `interface/relatorio.py` |
| Onde está a alternativa antiga de rede? | `treinamento_mobilenet.py`, `experimento_mobilenet.py` |

Os identificadores internos continuam minúsculos para compatibilidade com os pesos e
relatórios. A interface e as exportações de resultado aplicam maiúscula inicial.
