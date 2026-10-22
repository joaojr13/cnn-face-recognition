#!/bin/bash
# Executa a avaliação do modelo ativo e os testes; a saída fica em output/avaliacao.log.
cd "$(dirname "$0")" || exit 1
mkdir -p output
{
  echo "=== Avaliação ($(date)) ==="
  .venv/bin/python -m rostos.avaliacao_facial --aplicar-limiar
  echo "=== Código de saída da avaliação: $? ==="
  echo "=== Testes ==="
  .venv/bin/python -m pytest -q -p no:cacheprovider
  echo "=== Código de saída dos testes: $? ==="
} 2>&1 | tee output/avaliacao.log
echo "Concluído. Pode fechar esta janela."
