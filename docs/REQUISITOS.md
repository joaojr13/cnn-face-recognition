# Requisitos e implementação

O trabalho tem como objetivo reconhecer os cinco integrantes do grupo usando uma rede
neural convolucional. Os resultados incluem imagens do processamento, registro das fotos
de treinamento, matriz de confusão e métricas de avaliação. A data de entrega informada
pelo grupo é 22/11.

| Requisito | Implementação | Limitação atual |
|---|---|---|
| Reconhecimento com CNN | MTCNN localiza os rostos; InceptionResnetV1 extrai características; a camada final aprende os integrantes. | A base usa pesos pré-treinados congelados; somente a camada final é ajustada. |
| Região reduzida da imagem | Recorte do rosto com margem de 8%. | Parte do fundo pode permanecer dentro do retângulo. |
| Padronização | RGB 160 × 160, proporção preservada, preenchimento preto e normalização numérica. | Pose, iluminação e qualidade da fotografia ainda variam. |
| Imagens intermediárias | Original preparada, caixas, recorte, entrada e oito mapas reais de ativação. | Os mapas mostram respostas da rede; não explicam sozinhos a decisão. |
| Registro do treinamento | Manifesto com arquivos, rótulos, hashes e recortes. | A pasta da sessão não comprova a ocasião de captura. |
| Matriz e métricas | Diagnóstico das 10 fotos de treino do modelo ativo. | Teste independente e calibração do limiar ainda não concluídos. |
| Rejeição de externos | Resposta Desconhecido quando a pontuação fica abaixo do limiar. | Pessoas externas também podem receber pontuações altas. |

As orientações fornecidas ao grupo sugeriram suavização e equalização em caso de
dificuldade. Essas operações não foram incorporadas ao processamento atual. A estratégia
escolhida pelo grupo usa transferência de aprendizado; as informações recebidas não
registram confirmação explícita do professor sobre o uso de pesos pré-treinados.

O conjunto atual de treino tem duas fotos de Diego, Fernanda, Guilherme, Joao e Samanta.
As divisões de validação e teste ficam fora da pasta lida pelo treinamento experimental.
A base externa é separada por identidade, com pessoas diferentes nas duas divisões.

[Funcionamento do reconhecimento](COMO_FUNCIONA.md), [mapa do código](MAPA_DO_CODIGO.md)
e [validação e limitações](VALIDACAO.md) detalham o comportamento do programa e as evidências disponíveis.
