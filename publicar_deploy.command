#!/bin/bash
# Monta o pacote privado de deploy e envia para o repositório PRIVADO do GitHub.
# O envio é cancelado se o repositório for público (o pacote contém fotos dos integrantes).
cd "$(dirname "$0")" || exit 1
mkdir -p output
.venv/bin/python scripts/preparar_deploy.py \
  --remote https://github.com/joaojr13/cnn-face-recognition-config.git --push --substituir-remoto 2>&1 | tee output/deploy.log
echo "Concluído. Pode fechar esta janela."
