#!/bin/zsh
cd "$(dirname "$0")" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  echo 'Crie o ambiente seguindo o README antes de iniciar.'
  exit 1
fi
exec .venv/bin/python -m streamlit run app.py
