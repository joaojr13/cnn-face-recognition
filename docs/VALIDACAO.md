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

As fotos de teste dos integrantes estão reservadas. As pastas de validação dos integrantes
ainda não contêm imagens, e o limiar de 90% não foi calibrado. A base externa preparada
contém 50 fotos para validação e 50 para teste, com identidades diferentes entre as divisões.
O reconhecimento ainda não foi executado nesse conjunto externo. Não há uma taxa medida
de falsos aceites nem resultado final de teste para o modelo facial ativo.

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

Naquela migração, a base tinha 13 fotos e o modelo não foi retreinado. `MIGRACAO.json`
registra os hashes desse estado histórico. Depois, o usuário reorganizou a base e um novo
treinamento utilizou as 10 fotos atuais. O registro de migração não descreve essa nova base.

A preparação da base externa passou por 18 testes selecionados de integridade, separação
e interface. Após o treinamento mais recente, quatro testes de interface passaram;
também foram conferidos o novo modelo, sua galeria e o relatório de reconhecimento.
Esses números documentam as verificações já executadas, sem representar uma taxa de acerto facial.

O upload de uma foto de grupo foi verificado no navegador. As exportações PNG, TXT e JSON
foram verificadas pela execução de `main.py`. A captura física da webcam nesta versão,
a construção do container Docker e o deploy externo ainda não foram testados.
