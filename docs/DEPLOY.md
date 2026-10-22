# Deploy com Streamlit

O projeto tem um único ponto de entrada web, `app.py`. Não requer FastAPI, Uvicorn,
compilação JavaScript ou serviço separado para reconhecimento. O modelo roda no servidor,
em CPU; os navegadores dos integrantes enviam fotos pelo Streamlit.

O repositório público não inclui fotos, modelos treinados nem relatórios individuais.
Esses arquivos permanecem locais e devem ser fornecidos por um canal privado para
uma implantação com reconhecimento. Não devem ser adicionados ao Git público para deploy.

## Streamlit Community Cloud

1. Disponibilize esta pasta em um repositório que a sua conta do serviço possa acessar.
   Não inclua `.venv`, `.python`, `.cache` ou `output`.
2. Em um ambiente de acesso restrito, forneça `artifacts/active.json`, a pasta da execução ativa com `model.pt` e `metadata.json`,
   os relatórios dessa execução e `data/raw/` para permitir a galeria. O modelo ativo tem
   aproximadamente 90 MiB; transfira o arquivo completo, não um ponteiro sem os pesos.
3. No serviço, selecione o repositório e a branch, use `app.py` como entrada e **Python 3.11**.
4. Use `requirements.txt` da raiz. Ele instala TensorFlow, PyTorch e demais dependências;
   o primeiro provisionamento é maior que o da AP01. O arquivo de versões macOS é apenas um registro local.
5. Após subir, teste uma foto individual, uma de grupo, a galeria e a avaliação. Teste
   também a webcam usando o endereço HTTPS. Configure a visibilidade conforme o acesso desejado.

O projeto foi preparado para esse fluxo; **a publicação externa ainda não foi realizada**.
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
