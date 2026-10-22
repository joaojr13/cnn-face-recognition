"""Conteúdo explicativo: funcionamento, finalidade e interpretação dos resultados."""
INTRO = """O programa analisa uma fotografia para prever quais integrantes do grupo aparecem nela.
Primeiro localiza os rostos. Depois prepara cada rosto separadamente, calcula uma pontuação para
cada nome cadastrado e compara a maior pontuação com um valor de corte, chamado **limiar**.
Quando a pontuação fica abaixo desse valor, o resultado é **Desconhecido**.

O reconhecimento usa um modelo já treinado. Enviar uma foto ou capturá-la pela webcam
não altera esse modelo nem acrescenta a imagem ao treinamento."""

TREINAMENTO = """**No treinamento, o programa aprende a associar imagens aos nomes dos integrantes.**
Cada foto tem um rótulo: o nome da pessoa, obtido da pasta onde o arquivo foi organizado.
A rede recebe os pixels do rosto; o nome da pasta fornece a resposta correta para calcular o erro.

A primeira parte da rede já foi treinada com fotografias de outras pessoas. Ela transforma
cada rosto em 512 números que representam padrões encontrados na imagem. Essa parte fica
**congelada**, ou seja, seus parâmetros não mudam durante o treinamento do grupo.
Somente a camada final aprende a associar esses números aos integrantes.
Reaproveitar a primeira parte da rede é chamado de **transferência de aprendizado**.

O treinamento cria pequenas variações das fotos, como espelhamento e mudanças de brilho,
para expor a camada final a mudanças semelhantes às que podem acontecer em novas imagens.
São nove entradas derivadas de cada original, incluindo a própria imagem padronizada.
Elas continuam sendo versões da mesma foto, não nove observações independentes.

**No reconhecimento, o programa aplica o que já foi aprendido.** Uma foto passa pela mesma
preparação, a rede calcula as pontuações e o limiar decide se o nome será aceito.
Os parâmetros permanecem fixos durante essa análise."""

PASSOS = [
("Etapa 1 · Preparação da fotografia", """O arquivo enviado contém uma imagem formada por pequenos pontos de cor, chamados
**pixels**. O programa verifica o formato e o tamanho do arquivo e usa a informação de
orientação da câmera, quando disponível, para corrigir fotografias giradas.

A imagem é convertida para **RGB**: cada pixel passa a ter três valores, correspondentes
a vermelho, verde e azul. Se o maior lado ultrapassar 1.600 pixels, a foto é reduzida
proporcionalmente. Isso limita o trabalho necessário para localizar os rostos.

O resultado desta etapa é a fotografia preparada que aparece abaixo. São aceitos JPEG,
PNG e WebP, com até 12 MB e 24 megapixels."""),
("Etapa 2 · Localização dos rostos", """O programa procura regiões que tenham aparência de rosto e desenha um retângulo ao
redor de cada uma. A **MTCNN** é o detector usado: três pequenas redes convolucionais
propõem regiões, descartam candidatas fracas e refinam os retângulos.

O detector atribui uma pontuação a cada região. O programa mantém as detecções com valor
pelo menos 0,95 que também atendem aos limites de tamanho. Essa pontuação se refere à
presença de um rosto; a escolha do nome acontece nas etapas seguintes.

Os rostos recebem números da esquerda para a direita. Cada retângulo fornece as
coordenadas usadas para recortar a pessoa na próxima etapa. Rostos muito pequenos,
pouco iluminados ou parcialmente encobertos podem não ser localizados."""),
("Etapa 3 · Recorte da região do rosto", """A partir do retângulo detectado, o programa separa a região que será enviada à rede de
reconhecimento. O recorte inclui uma margem de 8% da maior dimensão do retângulo em cada
lado, limitada pelas bordas da fotografia. Essa margem reduz o risco de cortar detalhes próximos ao rosto.

Assim, grande parte da roupa, dos objetos e do cenário deixa de fazer parte da entrada.
Ainda pode restar fundo ao redor da pessoa: esta operação recorta uma região retangular,
sem apagar o fundo ou reposicionar os olhos. A imagem abaixo é o recorte obtido nesta execução."""),
("Etapa 4 · Padronização do tamanho", """Os rostos recortados podem ter tamanhos e proporções diferentes. A rede facial usada no
projeto espera sempre uma imagem de **160 × 160 pixels com três canais de cor**.

O programa redimensiona o recorte preservando sua proporção e preenche de preto o espaço
que sobra. Isso produz o tamanho esperado sem esticar o rosto. A interpolação Lanczos
calcula os valores dos pixels durante a mudança de tamanho.

A imagem abaixo representa a entrada padronizada. Tanto as fotos de treinamento quanto
as fotos analisadas na aplicação passam por essa mesma preparação."""),
("Etapa 5 · Normalização dos pixels", """Cada canal de cor da imagem tem valores de 0 a 255. Antes de executar a CNN facial,
o programa transforma cada valor pela fórmula **(valor − 127,5) / 128**. Essa é a escala
numérica esperada pela rede pré-treinada.

Por exemplo, o valor 0 se torna aproximadamente −0,996, e o valor 255 se torna
aproximadamente +0,996. A transformação mantém os três canais de cor e suas diferenças,
mas representa os pixels com números próximos do intervalo de −1 a 1.

O quadro abaixo usa o pixel real da posição x=80, y=80 do rosto selecionado. Ele permite
comparar os três valores RGB com os números que serão usados pela rede."""),
("Etapa 6 · Extração de características pela CNN", """Uma **rede neural convolucional (CNN)** processa a imagem usando filtros aprendidos.
Cada filtro combina valores de pequenas regiões da imagem e produz respostas mais fortes
para certos padrões. As camadas seguintes combinam essas respostas em representações mais complexas.

Neste projeto, a rede InceptionResnetV1 foi pré-treinada na base de rostos VGGFace2.
Ela transforma cada rosto em **512 números**, chamados de características. Esses números
serão usados pela camada final para calcular as pontuações dos integrantes.

As imagens abaixo são **mapas de ativação**: mostram respostas reais de oito canais da
primeira camada nesta análise, após os ajustes internos chamados BatchNorm e ReLU.
Cada canal foi convertido separadamente para uma escala de cinza. Uma região clara
indica uma resposta maior naquele canal; o brilho de canais diferentes não é diretamente comparável.
Esses mapas ajudam a observar o processamento, mas não indicam sozinhos por que um nome foi escolhido."""),
("Etapa 7 · Pontuação de cada integrante", """A camada final recebe as 512 características do rosto e calcula um valor para cada nome
cadastrado. Seus parâmetros foram ajustados durante o treinamento com as fotos dos integrantes.
Uma função chamada **softmax** transforma esses valores em pontuações que somam 100%.

As barras abaixo mostram a distribuição das pontuações nesta execução. O nome com a
maior pontuação é o candidato escolhido pela rede. Ele ainda precisa passar pelo limiar.

Esses percentuais comparam os integrantes cadastrados. Uma pontuação de 90% não significa
90% de certeza de que a identidade está correta. Mesmo uma pessoa externa recebe uma
distribuição entre os nomes conhecidos, pois são essas as saídas disponíveis na rede."""),
("Etapa 8 · Decisão pelo limiar", """O programa compara a maior pontuação com o **limiar**, que é o valor mínimo necessário
para aceitar o nome. Se a pontuação atingir ou superar esse corte, o nome candidato é
exibido. Se ficar abaixo, a previsão é recusada e o resultado passa a ser **Desconhecido**.

A comparação abaixo usa a pontuação e o limiar desta execução. A distância ao limiar
é a diferença entre os dois valores, em pontos percentuais: um valor negativo indica rejeição.

Desconhecido é uma regra de rejeição, não uma pessoa aprendida pela rede. Um integrante
pode ser rejeitado, e uma pessoa externa pode receber uma pontuação alta e ser aceita.
O limiar do modelo facial atual ainda é provisório: a calibração e a avaliação independente
não foram concluídas. Diminuir o corte pode aceitar mais integrantes e também mais pessoas externas."""),
]

AVALIACAO = """A avaliação compara a resposta do modelo com o nome correto de cada foto. O significado
dos números depende de **quais imagens foram usadas nessa comparação**.

- **Treinamento:** as fotos ajustam os parâmetros da camada final. O diagnóstico nessas
  mesmas fotos mostra como o modelo se ajustou aos exemplos que recebeu.
- **Validação:** fotos separadas do treino ajudam a escolher ajustes, como o limiar de
  rejeição, levando em conta integrantes e pessoas externas.
- **Teste:** outras fotos medem o resultado depois de definir o modelo e os ajustes.
  Quando os resultados de teste passam a orientar ajustes, deixam de ser uma avaliação independente dessas escolhas.

Ter arquivos em pastas de validação e teste não significa que essa avaliação já ocorreu.
As seções abaixo identificam quais resultados estão disponíveis para o modelo ativo.
Versões editadas de uma mesma foto são relacionadas; distribuí-las entre treino e teste
poderia produzir resultados excessivamente otimistas."""

METRICAS = """Na **matriz de confusão**, cada linha corresponde à pessoa correta e cada coluna à resposta
do modelo. Cada número conta quantas fotos tiveram aquela combinação. A diagonal reúne
os acertos de nome; as outras células mostram trocas de identidade e rejeições.

Por exemplo, a célula na linha **Samanta** e na coluna **Desconhecido** conta as fotos de
Samanta que o modelo recusou. A linha Desconhecido só mede o comportamento com pessoas
externas quando a avaliação inclui fotos externas; uma linha zerada por falta de exemplos não comprova acerto.

| Medida | Cálculo | O que significa |
|---|---|---|
| Acurácia | acertos / total | Fração das fotos com resposta correta. |
| Precisão de uma pessoa | VP / (VP + FP) | Entre as previsões com esse nome, a fração que estava correta. |
| Recall de uma pessoa | VP / (VP + FN) | Entre as fotos dessa pessoa, a fração reconhecida corretamente. |
| F1 | 2 × precisão × recall / (precisão + recall) | Combina precisão e recall; cai quando uma delas é baixa. |
| Suporte | quantidade de fotos da pessoa | Número de exemplos usados para calcular a medida. |
| Média macro | média simples das medidas por pessoa | Dá o mesmo peso às pessoas incluídas na média; no diagnóstico de treino, entram aquelas com suporte. |

**VP** é uma foto da pessoa reconhecida com o nome correto. **FP** é uma foto de outra
pessoa que recebeu esse nome. **FN** é uma foto da pessoa que foi confundida ou rejeitada.
Exemplo ilustrativo: entre quatro fotos de uma pessoa, três recebem o nome correto e uma
é rejeitada. Se mais uma foto de outra pessoa receber esse nome por engano, precisão e
recall serão 3/4, e F1 será 0,75. Se não houver nenhum acerto, o código representa F1 como zero.

Uma taxa de 100% em uma única foto tem pouco suporte. Quando há muito mais desconhecidos
que integrantes, a acurácia geral também pode parecer alta mesmo com muitas rejeições dos
integrantes. Por isso, as medidas por pessoa e a quantidade de exemplos ajudam na interpretação.
Um traço na tabela indica ausência de exemplos para aquela pessoa.

A **taxa de falsos aceites de desconhecidos** é a fração das fotos externas que receberam
indevidamente o nome de um integrante. O diagnóstico nas fotos de treino dos integrantes
não calcula essa taxa, pois não contém pessoas externas. A base externa preparada só
fornecerá essa medida quando for efetivamente avaliada."""

VISAO_GERAL = """O projeto transforma uma fotografia em previsões de identidade para os rostos encontrados.
A imagem inteira serve para localizar os rostos; cada região recortada é analisada separadamente
pela rede de reconhecimento. O resultado pode ser um nome do grupo ou a rejeição Desconhecido.

| Parte do programa | O que acontece | Finalidade |
|---|---|---|
| Detecção | A MTCNN localiza regiões de rosto e fornece retângulos. | Separar as pessoas de uma foto individual ou de grupo. |
| Preparação | Cada região é recortada, ajustada para 160 × 160 RGB e normalizada. | Fornecer o mesmo formato usado no treinamento. |
| CNN facial | A base pré-treinada extrai 512 características de cada rosto. | Representar padrões da imagem em números. |
| Camada final | Os parâmetros treinados associam características aos integrantes. | Calcular uma pontuação por nome. |
| Decisão | A maior pontuação é comparada com o limiar. | Aceitar o candidato ou retornar Desconhecido. |
| Avaliação | Previsões são comparadas com os rótulos corretos. | Medir acertos, confusões e rejeições no conjunto avaliado. |

O passo a passo do reconhecimento usa imagens e valores da própria execução.
A galeria de treinamento usa o registro de arquivos do modelo ativo. Assim, é possível
relacionar o resultado aos dados utilizados, ao preparo aplicado e à regra de decisão."""
