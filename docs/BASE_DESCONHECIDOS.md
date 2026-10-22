# Base externa de rostos — LFW

Preparada em 21/10/2026 para estudar a rejeição de pessoas externas aos cinco integrantes.
**100 fotografias de 50 identidades**, com duas fotos por pessoa, divididas por identidade:

| Divisão | Pessoas | Fotos | Uso |
|---|---:|---:|---|
| Validação | 25 | 50 | Escolher o limiar junto das fotos de validação dos integrantes |
| Teste | 25 | 50 | Medir falsos aceites depois de fixar modelo e limiar |

Não houve treinamento, reconhecimento de identidade ou ajuste do limiar na montagem.
As fotos não vão para `data/raw/`: Desconhecido continua sendo uma rejeição da saída,
não uma sexta identidade ensinada ao classificador.

## Arquivos

```text
data/unknown/
  manifest.json
  validation/lfw_<identificador>/foto_01.jpg, foto_02.jpg
  test/lfw_<outro_identificador>/foto_01.jpg, foto_02.jpg
```

O identificador da pasta é derivado do rótulo da identidade na base LFW. O manifesto
preserva o rótulo fornecido pelo dataset, o nome original do arquivo, a divisão, o hash
SHA256, as dimensões, a caixa detectada e a pontuação de detecção. Esses nomes são
metadados de origem; não foram inferidos pelo modelo deste projeto.

## Como a amostra foi montada

1. Baixar o arquivo **LFW original**, sem o alinhamento adicional da variante funneled.
2. Conferir o SHA256 do arquivo com a referência usada pelo scikit-learn.
3. Embaralhar identidades com pelo menos duas fotos usando semente 42; embaralhar também
   as fotos de cada identidade. Não selecionar por nome, aparência ou pontuação de reconhecimento.
4. Percorrer essa ordem e aceitar apenas arquivos válidos com exatamente um rosto
   detectado pela MTCNN do projeto (corte de detecção 0,95). Exclusões e motivos ficam no manifesto.
5. Selecionar 50 identidades com duas fotos utilizáveis cada e embaralhá-las novamente
   com semente 43 antes de separar 25 para validação e 25 para teste.
6. Manter os bytes JPEG originais; gerar recorte e entrada 160 × 160 apenas para inspeção
   ou processamento posterior. Conferir duplicatas por SHA256 e ausência de repetição de
   identidade entre as divisões. Conferir também duplicatas contra as fotos dos integrantes.

A CNN de identidade não é chamada para escolher as fotos. Não foram escolhidos somente
exemplos que o modelo atual rejeita corretamente. Preparar a imagem e conferir sua detecção
não é consumir previsões de teste para escolher o limiar.

## Consultar e reproduzir

A seção **Base de treinamento → Pessoas externas: validação e teste** contém o inventário
da base externa. Sua galeria mostra exemplos da validação, sem executar reconhecimento.
O teste permanece em sua pasta, reservado à avaliação final.

```sh
.venv/bin/python scripts/prepare_unknown_faces.py --verify
```

Para reproduzir em uma cópia do projeto com `data/unknown/` vazia:

```sh
.venv/bin/python scripts/prepare_unknown_faces.py
```

O comando baixa aproximadamente 172 MB para `.cache/datasets/lfw.tgz`, confere o arquivo
e adiciona somente o subconjunto à base. Se já houver um manifesto, verifica e preserva
os dados; não sobrescreve uma divisão existente. A reprodução das caixas pressupõe as
mesmas versões do detector e das dependências. A seleção está documentada no manifesto.

## Como usar com as novas fotos dos integrantes

As duas novas fotos por integrante solicitadas pelo grupo podem ser incorporadas ao
**treino**, preservando as originais existentes. Ainda serão necessárias fotos dos
integrantes reservadas para **validação** e **teste**, de preferência em ocasiões distintas.
Se só houver duas fotos novas no total por integrante, uma alternativa metodológica é
manter as fotos antigas no treino e reservar uma nova para validação e outra para teste;
isso deve ser decidido antes de retreinar. Uma foto por divisão é um diagnóstico pequeno,
não uma estimativa robusta de desempenho.

Para calibrar o limiar, precisamos observar tanto a aceitação dos integrantes quanto a
rejeição dos externos na validação. Usar apenas externos favoreceria rejeitar todo mundo.
Depois de fixar o modelo e o limiar, avaliar o teste sem reajustá-los pelos seus resultados.
Não misturar fotos de treino dos integrantes com externos de teste e chamar o resultado
combinado de avaliação independente.

O fluxo FaceNet atual ainda é experimental. Montar a base externa não integra automaticamente
a calibração e o teste ao treinamento facial; isso será feito quando as divisões conhecidas
estiverem definidas. A alternativa antiga MobileNet já lê `data/unknown`, mas não é o modelo ativo.

## Limitações

- A amostra foi condicionada a detecção bem-sucedida. Ela avalia rejeição **dado um rosto
  detectado**, não a taxa de falhas do detector em todo o LFW.
- LFW reúne fotografias de pessoas públicas e não representa necessariamente a iluminação,
  resolução, enquadramento ou diversidade das fotos dos integrantes e da webcam.
- Externo significa que a pessoa não é um dos cinco integrantes. Não há garantia de que
  suas imagens ou sua identidade nunca apareceram no pré-treinamento VGGFace2 da CNN.
- Duas fotos de uma identidade são relacionadas; 50 fotos de teste não são 50 identidades.
  Não há garantia de sessões independentes dentro de cada identidade LFW.
- A seleção não foi estratificada por características demográficas. Não foram inferidos
  atributos sensíveis para selecionar ou categorizar as pessoas.
- Um resultado bom nesta amostra não garante rejeição de qualquer pessoa externa.

## Origem e créditos

Gary B. Huang, Manu Ramesh, Tamara Berg e Erik Learned-Miller. *Labeled Faces in the Wild:
A Database for Studying Face Recognition in Unconstrained Environments*. University of
Massachusetts Amherst, Technical Report 07-49, 2007.

- [Descrição do LFW na documentação scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html)
- [Arquivo LFW no espelho Figshare, publicado por Nelson Liu](https://figshare.com/articles/dataset/lfw_tgz/3829986)
- [Referência do endereço e SHA256 no código do scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py)

O espelho Figshare informa CC BY 4.0. Essa indicação do depósito não constitui uma
verificação individual dos direitos sobre cada fotografia original. Este subconjunto foi
preparado para o experimento acadêmico local; nenhuma imagem foi publicada externamente
por esta operação. Preserve a procedência e os créditos junto da base.
