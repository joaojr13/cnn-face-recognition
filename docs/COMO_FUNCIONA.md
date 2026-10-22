# Como o reconhecimento de rostos funciona

O programa recebe uma fotografia, localiza os rostos e calcula uma previsão para cada um.
Os nomes disponíveis são Diego, Fernanda, Guilherme, Joao e Samanta. Quando a maior
pontuação não atinge o valor de corte configurado, a resposta é Desconhecido.

Uma fotografia com várias pessoas gera várias análises: cada rosto tem seu próprio recorte,
suas pontuações e sua decisão. A numeração dos rostos segue a posição da esquerda para a direita.

## Da fotografia ao rosto preparado

Uma imagem digital é uma grade de pixels. Na representação RGB, cada pixel contém três
valores de intensidade, um para vermelho, um para verde e outro para azul.

A leitura confere o arquivo, corrige a orientação indicada pela câmera e converte a imagem
para RGB. Fotografias maiores são reduzidas proporcionalmente para até 1.600 pixels no
maior lado, limitando o trabalho da etapa seguinte.

A MTCNN procura regiões com aparência de rosto. Suas três redes convolucionais propõem
regiões e refinam os resultados. O programa mantém detecções com pontuação pelo menos
0,95 e dimensões válidas. Cada região é representada por um retângulo. Nessa etapa,
o programa conhece a localização de um possível rosto, mas ainda não escolheu seu nome.

O recorte acrescenta uma margem de 8% da maior dimensão do retângulo em cada lado,
respeitando os limites da fotografia. Isso conserva detalhes próximos ao rosto e reduz
a quantidade de roupa e cenário enviada à rede. Parte do fundo pode permanecer no recorte.

O rosto é ajustado para 160 × 160 pixels, mantendo sua proporção. As áreas que sobram
são preenchidas de preto. A interpolação Lanczos calcula os pixels durante a mudança
de tamanho. A mesma preparação é aplicada no treinamento e no reconhecimento.

## A imagem como entrada numérica

Os valores RGB começam entre 0 e 255. A CNN facial espera outra escala, então cada canal
passa pela transformação `(valor − 127,5) / 128`. O resultado fica aproximadamente
entre −1 e 1. As três cores continuam presentes; a transformação muda a escala dos números.

Uma entrada tem 160 linhas, 160 colunas e três canais. Antes de executar a CNN, esses
valores são organizados no formato esperado pelo PyTorch: quantidade de rostos × canais
× altura × largura. Esse conjunto de números organizado por dimensões é chamado de tensor.

O passo a passo mostra um pixel real antes e depois da normalização. As miniaturas
são prévias em JPEG; a CNN usa os pixels preparados antes da compressão dessas prévias.
O hash SHA256 exibido identifica os pixels usados e permite conferir se a entrada mudou.

## O que a rede convolucional calcula

Uma rede neural convolucional, ou CNN, tem filtros que combinam valores de pequenas
regiões da imagem. Os números de cada filtro são parâmetros aprendidos no treinamento.
Ao percorrer a imagem, um filtro produz um mapa de respostas a padrões locais.
As camadas seguintes combinam essas respostas para formar representações mais complexas.

A rede utilizada é a InceptionResnetV1, pré-treinada com a base de rostos VGGFace2.
Ela produz 512 números por rosto. Esses números representam características calculadas
pela rede; não correspondem a uma lista de partes do rosto nomeadas manualmente.

A interface mostra oito mapas da primeira camada na mesma execução que produz a previsão.
Os mapas são capturados depois de BatchNorm e ReLU. BatchNorm ajusta a escala das respostas
usando estatísticas aprendidas; ReLU mantém os valores positivos e transforma os negativos
em zero. Cada canal é convertido separadamente para cinza para permitir sua visualização.
Uma região mais clara indica maior resposta dentro daquele canal. Isso não explica,
sozinho, por que o programa escolheu determinada identidade.

## Como o treinamento aprende os integrantes

As pastas das fotografias fornecem os rótulos, isto é, os nomes corretos. A rede recebe os
pixels preparados. O nome da pasta é usado para calcular o erro da resposta, sem ser
fornecido como parte da imagem.

A base pré-treinada permanece congelada: seus parâmetros não são atualizados com as fotos
do grupo. Somente a camada final é treinada para associar as 512 características aos cinco
integrantes. Essa camada tem 512 × 5 pesos e cinco valores de ajuste adicionais, totalizando
2.565 parâmetros treináveis. Esse reaproveitamento é chamado de transferência de aprendizado.

Durante o treinamento, a entropia cruzada calcula uma perda a partir da resposta e do rótulo
correto. Uma perda maior indica pior ajuste ao objetivo de treinamento. O otimizador AdamW
usa essa informação para atualizar os parâmetros da camada final. Cada época corresponde
a uma passagem de otimização pelas características usadas no treino. Como a base é fixa,
suas características podem ser calculadas uma vez e reutilizadas nas épocas seguintes.

O dropout desativa aleatoriamente parte das entradas da camada final durante o treino,
para reduzir a dependência de características específicas. Ele não atua no reconhecimento.

Cada foto origina nove entradas: original padronizada, espelhamento, tons de cinza,
rotações de −4° e +4°, mudanças de brilho de −15% e +15% e mudanças de contraste de −15%
e +15%. Essas versões expõem o treinamento a pequenas alterações. Continuam relacionadas
a uma única foto original e não representam novas pessoas ou ocasiões de captura.

Depois do treino, os parâmetros são salvos e carregados novamente para conferir a
consistência das previsões. O arquivo salvo passa a ser o modelo ativo. Enviar uma nova
foto ao site executa esse modelo, sem modificar seus parâmetros.

## Das características ao nome previsto

A camada final calcula um valor para cada integrante. A função softmax transforma esses
valores em pontuações que somam 100%. O nome com a maior pontuação é o candidato.

O programa aceita esse candidato apenas se sua pontuação atingir o limiar configurado.
O limiar ativo é 94,8%. Em um exemplo ilustrativo, uma pontuação de 94,0% resulta em
Desconhecido: a diferença é de −0,8 ponto percentual, por isso não passa pelo corte.

### Como o limiar de 94,8% foi escolhido

Um limiar baixo aceita integrantes, mas também pessoas externas com o nome de alguém do
grupo. Um limiar alto barra externos, mas recusa fotos dos integrantes. A escolha usou só
a validação e uma regra fixada antes do teste. Primeiro, aceitar no máximo 5% dos externos.
Depois, entre esses limiares, ficar com o de maior F1 macro, a média das cinco pessoas e
de Desconhecido. Em caso de empate, vence o menor limiar.

Na validação, o integrante com menor pontuação teve 0,906 (Samanta). O externo com maior
pontuação teve 0,948, para Guilherme. Como o externo ficou acima, nenhum limiar separa
tudo sem erro:

| Limiar | Integrantes aceitos | Externos aceitos | F1 macro |
|---|---|---|---|
| 0,900 (provisório) | 10/10 | 3/50 | 0,895 |
| 0,906 (aceita todos os integrantes) | 10/10 | 2/50 | 0,930 |
| **0,948 (calibrado)** | 9/10 | 0/50 | **0,943** |
| 0,990 | 9/10 | 0/50 | 0,943 |

No F1 macro, um externo aceito reduz a precisão do integrante cujo nome recebeu e o recall
de Desconhecido. Por isso, dois externos aceitos pesaram mais que uma foto recusada. Em 0,990,
o F1 empata, e o desempate fica com 0,948. No teste, que não participou da escolha, 0,948
aceitou os 5 integrantes e nenhum dos 50 externos. O provisório de 0,90 teria aceitado um externo.
A página Avaliação mostra essas curvas e permite simular outros limiares.

Desconhecido não é uma sexta identidade aprendida pela rede: é a recusa da previsão.
Um integrante pode ser recusado por pose, iluminação, diferenças em relação às fotos
de treino ou outros fatores. Uma pessoa externa também pode receber uma pontuação alta
para algum integrante. A pontuação é relativa aos nomes cadastrados e não foi calibrada
como probabilidade de identidade correta.

## O que a avaliação permite concluir

As métricas comparam previsões com rótulos conhecidos. A matriz de confusão reúne as
contagens: linhas indicam o nome correto e colunas indicam a resposta. A diagonal reúne
acertos; as outras células mostram confusões e rejeições.

Precisão mede quantas previsões de um nome estavam corretas. Recall mede quantas fotos
da pessoa foram reconhecidas. F1 combina essas duas medidas. Acurácia mede a fração
total de respostas corretas. O suporte informa quantas fotos sustentam cada resultado.

As fotos de treino ajustam o modelo. As fotos de validação permitem escolher ajustes,
como o limiar, usando integrantes e externos. As fotos de teste avaliam o modelo depois
de fixar essas escolhas. Usar uma foto para ajustar o sistema e depois tratá-la como
avaliação independente produz uma conclusão excessivamente otimista.

O modelo facial atual foi treinado com 10 fotos, duas de cada integrante. As 90 entradas
com variações serviram ao treino; o diagnóstico usa as 10 originais.

Sem fotos de validação dos integrantes, a validação usa *leave-one-out*: cada foto de treino
é prevista por uma camada final treinada sem ela. Essas previsões e 50 fotos LFW de validação
definiram o limiar de 0,948. O teste usou uma foto nova de cada integrante e 50 fotos LFW de
outras pessoas, sem participar da escolha do limiar. Os resultados estão em
[Validação e limitações](VALIDACAO.md) e na página Avaliação.

## Por que o resultado pode falhar

| Situação observada | Relação com o processamento |
|---|---|
| Nenhum retângulo de rosto | O detector não encontrou uma região que atendesse aos critérios; tamanho, pose e luz podem afetá-lo. |
| Parte do rosto ausente no recorte | A caixa detectada e a margem determinam os pixels que chegam à rede. |
| Nome candidato correto, mas resposta Desconhecido | A maior pontuação ficou abaixo do limiar. |
| Nome incorreto com pontuação alta | A rede pode confundir pessoas mesmo quando sua saída favorece fortemente um nome. |
| Acertos no treino e erros em novas fotos | Ajustar-se aos exemplos de treino não garante reconhecer variações não vistas. |
| Foto nova na pasta, mas ausente da galeria de treino | A galeria segue o registro da execução salva; a inclusão de arquivos não treina o modelo automaticamente. |

## Bibliotecas e operações utilizadas

Pillow e NumPy leem e manipulam os pixels. TensorFlow e MTCNN executam a detecção de rostos.
PyTorch e facenet-pytorch implementam a rede facial e a camada final. Streamlit organiza
as telas; pandas e Matplotlib exibem tabelas e gráficos. A alternativa de avaliação em
`treinamento_mobilenet.py` usa scikit-learn para métricas e calibração, mas essa alternativa
não corresponde ao modelo FaceNet ativo.

O tratamento atual consiste em leitura, recorte, redimensionamento e normalização.
Suavização adicional, equalização de histograma, alinhamento pelos olhos e transformada
de Fourier não integram esse fluxo. Alterar a preparação mudaria os dados fornecidos
à rede e exigiria verificar sua consistência e seu efeito no reconhecimento.
