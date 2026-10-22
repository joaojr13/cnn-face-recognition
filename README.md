# AP02 · Reconhecimento de rostos

Aplicação acadêmica em **Streamlit**, organizada como a AP01: `app.py` cuida da página,
`main.py` permite executar sem navegador, `rostos/pipeline.py` conduz o processamento
e `interface/` reúne componentes e explicações.

A versão incluída usa MTCNN para detecção e InceptionResnetV1 pré-treinada em VGGFace2
para extrair características. A base convolucional fica congelada e uma camada neural
final foi treinada para Diego, Fernanda, Guilherme, Joao e Samanta.

**Estado real:** 10 fotos originais (duas por integrante), modelo experimental, limiar provisório de 90%, sem
teste independente. Matriz de confusão e métricas disponíveis são diagnóstico das fotos
de treinamento. Elas não medem generalização. A migração não retreinou nem alterou os pesos.

## Conteúdo do repositório

Este repositório público contém código, documentação e estrutura de pastas.
Fotos, modelos treinados e relatórios individuais não são versionados. Os resultados descritos abaixo pertencem à execução local.
Para reconhecer rostos após clonar o repositório, é necessário fornecer as fotos
e treinar o modelo, ou instalar os artefatos autorizados por um canal privado.

## Executar

Use Python **3.11**. A partir desta pasta:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Se usar `uv`, substitua a criação e instalação por:

```sh
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

Abra o endereço exibido no terminal (normalmente http://localhost:8501).
No Mac, depois da instalação, também pode abrir `iniciar.command`.
O ambiente `.venv` instalado localmente não deve ser copiado para outro sistema operacional.

```sh
.venv/bin/python main.py /caminho/foto.jpg --output output/exemplo
```

O comando gera PNG anotado, resumo TXT e registro JSON. Não adiciona a imagem ao treino.
Os 10 arquivos da base, os pesos ativos e os relatórios existem na instalação local,
mas não acompanham o clone público.
Para apenas reconhecer não é necessário baixar os pesos VGGFace2 novamente.

## Organização

| Arquivo ou pasta | Responsabilidade |
|---|---|
| `app.py` | Entrada Streamlit e seleção de página |
| `main.py` | Entrada pela linha de comando |
| `rostos/pipeline.py` | Sequência de reconhecimento e proteção de concorrência |
| `rostos/imagem.py`, `deteccao.py` | Leitura, MTCNN, recorte e padronização |
| `rostos/etapas/preparacao.py` | Preparação por rosto e exemplo real de normalização |
| `rostos/modelo.py`, `treinamento_facial.py` | Modelo salvo, CNN facial, camada treinada e decisão |
| `rostos/saida.py` | Relatório auditável da previsão |
| `rostos/dados.py`, `arquivos.py`, `estudo.py` | Inventário, caminhos portáteis, manifesto e métricas |
| `interface/componentes.py`, `passos.py`, `textos.py` | Páginas, oito etapas e conteúdo didático |
| `interface/sessao.py`, `relatorio.py`, `nomes.py` | Estado por usuário, exportação e nomes em maiúscula inicial |
| `data/raw/` | Fotos organizadas por integrante e sessão |
| `artifacts/` | Modelo ativo e metadados |
| `reports/` | Manifesto, perda e diagnóstico da execução ativa |
| `docs/` | Funcionamento, mapa do código, validação e deploy |
| `tests/` | Testes do algoritmo, relatórios, portabilidade e interface |

## Usar a aplicação

- **Reconhecer:** enviar uma foto, capturar pela webcam ou testar uma imagem da base;
  acompanhar cada rosto nas oito etapas e baixar o resultado.
- **Base de treinamento:** inspecionar as fotos do manifesto, seus recortes e variações.
- **Avaliação:** ver matriz, acurácia, precisão, recall, F1, suporte e curva de perda;
  distinguir diagnóstico de treino de teste independente.
- **Entenda o projeto:** entender o processamento, o treinamento, a decisão e a organização do código.

A webcam captura uma fotografia, não faz reconhecimento contínuo em vídeo. Em hospedagem,
a câmera exige contexto seguro (HTTPS); `localhost` funciona para desenvolvimento.
Uploads permanecem na sessão e não são gravados automaticamente na base. O servidor
contém as fotos de treinamento e os resultados salvos previamente. O link, por si só,
não controla acesso: configure a visibilidade no provedor se necessário.

## Treinar novamente, quando houver novas fotos

```sh
.venv/bin/python scripts/import_images.py /pasta/com/subpastas/por/pessoa --session sessao_02
.venv/bin/python -m rostos.treinamento_facial --epochs 200 --threshold 0.9 --activate
```

Esse comando executa **novo experimento FaceNet em toda a base**, sem teste independente.
Ele pode baixar os pesos pré-treinados no primeiro treinamento. Não é executado ao iniciar
o site. `--activate` troca o modelo somente depois de salvar e conferir sua restauração.
Sem essa opção, a nova execução fica salva e a ativa permanece.

Os módulos `treinamento_mobilenet.py` e `experimento_mobilenet.py` preservam a alternativa
anterior. O primeiro tem separação de treino/validação/teste, calibração e avaliação,
mas usa **MobileNetV2**, não o FaceNet ativo. Não confunda suas capacidades com avaliações
já realizadas no modelo facial. Não foram trazidos pesos de experimentos antigos.

## Estudar, validar e publicar

- [Como o reconhecimento funciona](docs/COMO_FUNCIONA.md)
- [Mapa do código](docs/MAPA_DO_CODIGO.md)
- [Validação e limitações](docs/VALIDACAO.md)
- [Deploy com Streamlit](docs/DEPLOY.md)

```sh
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

`requirements.txt` declara dependências portáveis. `requirements-lock-macos.txt`, quando
presente, registra o ambiente local validado; não é o arquivo de instalação da nuvem.

## Base de pessoas externas

Foi preparado um subconjunto LFW com **100 fotos de 50 pessoas**, separado em 25 pessoas
para validação (50 fotos) e outras 25 para teste (50 fotos), em `data/unknown/`.
A tela **Base de treinamento** mostra o inventário e exemplos da validação, separados
das imagens de treino dos integrantes. A preparação não altera o modelo nem o limiar.

Consulte [origem, seleção e uso da base externa](docs/BASE_DESCONHECIDOS.md).

```sh
.venv/bin/python scripts/prepare_unknown_faces.py --verify
```

## Onde colocar as duas novas fotos de cada integrante

As fotos atuais permanecem em `data/raw/<integrante>/sessao_01/` para treinamento.
Foram preparadas pastas separadas para as novas imagens:

- `data/validation/<integrante>/`: uma foto nova para validação.
- `data/test/<integrante>/`: outra foto nova para teste final.

As cinco pastas em cada divisão são `diego`, `fernanda`, `guilherme`, `joao` e `samanta`.
Use fotos diferentes das que participaram do treino; versões da mesma foto devem
permanecer na mesma divisão. Veja [as instruções da coleta](data/LEIA-ME.txt).
Essas pastas ficam fora da leitura do treinamento facial experimental, que continua
usando somente `data/raw`. A avaliação facial será integrada a elas após a coleta.

## Novo treinamento — 21/10/2026

Modelo ativo: `facenet_experimental_20261022T023553Z_53a142`. Treinamento refeito com 10 fotos,
duas por integrante, 90 entradas aumentadas e 200 épocas. Limiar provisório 0,90.
Cinco fotos estão reservadas em `data/test/`; `data/validation/` ainda não contém fotos
dos integrantes. O teste não foi executado e o limiar não foi calibrado. O diagnóstico
acertou 10/10 imagens de treinamento; isso não estima desempenho em novas fotos.
