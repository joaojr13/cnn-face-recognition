# Validação e limitações

Existem duas verificações diferentes no projeto. Os testes de software conferem se as
operações estão implementadas como esperado. A avaliação do reconhecimento compara
as previsões com os nomes corretos de pessoas em fotografias. Um programa pode passar
nos testes de software e ainda errar a identificação em fotos novas.

## Resultados disponíveis do modelo ativo

O treinamento facial mais recente utilizou 10 fotos originais, duas por integrante,
com nove entradas derivadas de cada uma e 200 épocas. O diagnóstico nas 10 originais
acertou os 10 nomes. Essas imagens participaram do treinamento, portanto o resultado
mede o ajuste à base, sem estimar o desempenho em fotografias novas.

A avaliação de 22/10/2026 (`rostos.avaliacao_facial`) calibrou o limiar em 0,948 na
validação: 10 previsões leave-one-out dos integrantes e 50 fotos LFW, com F1 macro 0,943,
nenhum externo aceito e 1 integrante rejeitado (Samanta, pontuação 0,906).

No teste independente, com o limiar calibrado, as 5 fotos novas receberam o nome correto
(pontuações de 0,966 a 0,999) e as 50 fotos externas foram rejeitadas: acurácia 100%,
F1 macro 1,000. Com o limiar provisório de 0,90, o mesmo teste teria aceitado um externo
como Joao (pontuação 0,936): acurácia 98,2%, 2% de falsos aceites. A validação cruzada
acertou as 15 fotos dos integrantes, com pontuação mínima de 0,954.

Esses números vêm de amostras pequenas. A distância entre o integrante com menor
pontuação no teste (0,966) e o externo com maior pontuação (0,936) é de 3 pontos, e o limiar
fica entre os dois. Fotos com outra iluminação ou pose podem cair abaixo do limiar.

### Avaliação sem fotos novas

Como não há mais fotos para treinar, `rostos/avaliacao_facial.py` usa as imagens existentes
em três papéis separados. O limiar é escolhido na validação: cada foto de treino é prevista
por uma camada final treinada sem ela (leave-one-out), junto com as 50 fotos LFW de
validação. O teste independente aplica o modelo ativo, sem alterações, às 5 fotos de
`data/test/` e às 50 fotos LFW de teste. A validação cruzada complementar prevê cada uma
das 15 fotos dos integrantes com uma camada treinada nas outras 14.

O teste tem uma foto por integrante, então o recall de cada pessoa só pode ser 0% ou 100%.
A validação cruzada dá três exemplos por pessoa, mas reutiliza as fotos de teste e não
mede a rejeição de externos. Os dois resultados devem ser apresentados juntos, com essas ressalvas.

As contagens atuais da execução são registradas em `artifacts/active.json` e nos relatórios
associados. A galeria reproduz as fotos desse registro, não todas as imagens que possam
ter sido adicionadas posteriormente às pastas.

## O que os testes de software verificam

- A leitura trata formatos inválidos, orientação da câmera, imagens auxiliares em JPEG e limites de tamanho.
- O recorte e a padronização preservam o formato usado tanto no treino quanto no reconhecimento.
- A decisão respeita o limiar; o relatório usa a mesma versão de modelo da previsão.
- A galeria confere o hash, que identifica o conteúdo do arquivo, e bloqueia caminhos fora da base.
- O cálculo das métricas conta confusões e rejeições e distingue ausência de exemplos de desempenho zero.
- A troca de foto ou modelo invalida resultados antigos; cada sessão mantém suas próprias imagens.
- A rede facial mantém a base congelada e produz previsões consistentes após salvar e restaurar os pesos.
- Os mapas de ativação vêm da mesma passagem pela rede que calculou as pontuações.
- A base externa impede arquivos alterados e repetição de identidade entre validação e teste.

Os comandos abaixo executam verificações do software. A opção `not ml` exclui os testes
que carregam redes reais. Os testes da alternativa MobileNet podem baixar pesos na primeira execução.

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -m 'not ml' -q
```

## Evidências registradas ao longo do desenvolvimento

Na migração de 21/10/2026 para Streamlit, 24 testes passaram, incluindo cinco integrações
com redes reais. Foram comparadas 14 imagens, contendo 18 rostos, entre a versão anterior
e a nova: caixas, decisões, pontuações, hashes de entrada e imagens intermediárias foram
preservados. A tolerância numérica das pontuações foi de 1e-6.

Naquela migração, a base tinha 13 fotos e o modelo não foi retreinado. Depois, o usuário
reorganizou a base e um novo treinamento utilizou as 10 fotos atuais.

A preparação da base externa passou por 18 testes selecionados de integridade, separação
e interface. Após o treinamento mais recente, quatro testes de interface passaram;
também foram conferidos o novo modelo, sua galeria e o relatório de reconhecimento.
Esses números documentam as verificações já executadas, sem representar uma taxa de acerto facial.

O upload de uma foto de grupo foi verificado no navegador. As exportações PNG, TXT e JSON
foram verificadas pela execução de `main.py`. A captura física da webcam nesta versão,
a construção do container Docker e o deploy externo ainda não foram testados.
