# Deploy com Streamlit

O projeto tem um único ponto de entrada web, `app.py`. Não requer FastAPI, Uvicorn,
compilação JavaScript ou serviço separado para reconhecimento. O modelo roda no servidor,
em CPU; os navegadores dos integrantes enviam fotos pelo Streamlit.

O repositório público não inclui fotos, modelos treinados nem relatórios individuais.
Esses arquivos permanecem locais e devem ser fornecidos por um canal privado para
uma implantação com reconhecimento. Não devem ser adicionados ao Git público para deploy.

## Qual limiar o app usa

O app lê o limiar de `artifacts/active.json`, junto com o modelo. Essa pasta está no
`.gitignore`. Publicado a partir do repositório público, o app não tem modelo e funciona
só como detector de rostos. O deploy precisa levar o modelo ativo, cujo limiar calibrado
é 0,948, e a avaliação dele.

## Streamlit Community Cloud: repositório privado de deploy

O repositório público continua sem fotos e sem pesos. Um segundo repositório, **privado**,
recebe o código e os arquivos de execução, montado por `scripts/preparar_deploy.py`. Antes
de copiar, o script confere:

- o modelo ativo, o `metadata.json` e o limiar são da mesma execução;
- o limiar está calibrado, e o `evaluation.json` é deste modelo e deste limiar;
- o `model.pt` tem os pesos reais (não um ponteiro) e cabe no limite de 100 MB do GitHub.

Ele copia: código, `artifacts/active.json`, a pasta do modelo ativo, os relatórios dessa
execução (com a avaliação e as figuras), `data/raw/` (galeria) e `data/unknown/` (base externa).
As fotos de `data/test/`, `.venv`, `.cache` e `output` não vão.

### Primeira publicação

1. No GitHub, crie um repositório **privado** e vazio (por exemplo, `cnn-face-recognition-deploy`).
2. Na pasta do projeto:
   ```sh
   .venv/bin/python scripts/preparar_deploy.py --remote https://github.com/<usuario>/cnn-face-recognition-deploy.git --push
   ```
   A cópia fica em `../ap-02-deploy-privado`, com um commit e um `DEPLOY_PRIVADO.md`.
   Se o repositório foi criado com README ou licença, acrescente `--substituir-remoto`: o pacote
   entra por cima desse commit, sem force push. Antes de enviar, o script consulta o GitHub e
   cancela o envio se o repositório for público.
3. No Streamlit Community Cloud, em **Settings → Linked accounts**, autorize o acesso a
   repositórios privados.
4. **Create app** → o repositório privado, branch `main`, arquivo `app.py`; em
   **Advanced settings**, escolha **Python 3.11**.
5. Depois do deploy, confira a barra lateral: "Limiar: 94.8% (calibrado na validação)".
   Teste uma foto individual, uma de grupo, a galeria, a página Avaliação e a webcam (HTTPS).
6. Em **Settings → Sharing**, restrinja quem pode ver o app. O app mostra as fotos de treino na galeria.

### Atualizar

Após mudar o código, treinar de novo ou recalibrar:

```sh
.venv/bin/python scripts/preparar_deploy.py --push
```

Arquivos removidos no projeto, como um modelo antigo, também saem da cópia. Um modelo novo
com limiar provisório é recusado até rodar `rostos.avaliacao_facial --aplicar-limiar`.

### Recursos

O plano gratuito oferece até cerca de 2,7 GB de memória. `requirements.txt` instala
TensorFlow (detector MTCNN) e PyTorch (FaceNet). O primeiro provisionamento é demorado
e o uso de memória deve ser conferido em **Manage app** após o deploy. Se faltar memória,
use a alternativa com Docker abaixo.

A publicação externa ainda não foi realizada.
A disponibilidade de memória/CPU do plano precisa ser verificada no provedor. Se houver falta
 de memória com TensorFlow e PyTorch, use um servidor/container com recursos suficientes.
A inferência é serializada para proteger as redes entre sessões, portanto uploads simultâneos aguardam a vez.

As fotos de treino acompanham o pacote autorizado pelo grupo. Uploads e capturas ficam
na sessão do usuário; não são usados para treinar nem persistidos automaticamente.
O código não inclui autenticação própria. Quem tiver acesso ao aplicativo poderá ver a galeria.
O filesystem de uma hospedagem pode ser temporário: mantenha os artefatos do treino no pacote
ou em armazenamento persistente, e faça novos treinamentos fora da interface web.

## Servidor com Docker (alternativa)

```sh
docker build -t ap02-rostos .
docker run --rm -p 8501:8501 ap02-rostos
```

A receita usa Python 3.11 e rodas CPU do PyTorch. O container escuta na porta 8501.
Para acesso externo com webcam, configure HTTPS no serviço/proxy. O Dockerfile é uma
receita de implantação; confira `VALIDACAO.md` para distinguir testes locais de testes em nuvem.

## Referências oficiais

- [Organização e publicação no Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [Dependências da aplicação](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Captura de fotografia com camera_input](https://docs.streamlit.io/develop/api-reference/widgets/st.camera_input)
- [Cache de recursos compartilhados e concorrência](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)
