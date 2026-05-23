"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

Executa quatro experimentos em sequência:
  1. Classificação multiclasse com as imagens originais (10 classes, imagens 28x28)
  2. Classificação binária com as imagens originais     (Camiseta vs Calça)
  3. Classificação multiclasse com HOG+LBP              (10 classes, vetores de descritores)
  4. Classificação binária com HOG+LBP                  (Camiseta vs Calça)

Divisão dos dados:
  - Treino:   60.000 amostras
  - Validação: 20% do treino via validation_split
  - Teste:    10.000 amostras

O Fashion MNIST já vem pré-dividido em 60k/10k pelo Keras, o que é a
divisão mais comum usada na literatura.
"""

import os
import sys
import numpy as np

# Garante que os módulos do projeto sejam encontrados
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow import keras

from cnn import CNN, CLASSES_BINARIAS
from descritores import extrair_hog_lbp


# ============================================================
# CONFIGURAÇÃO DOS EXPERIMENTOS
# ============================================================

# Classes da tarefa binária: Camiseta (0) vs Calça (1)
CLASSES_BINARIA = [0, 1]

EPOCAS          = 20       # Máximo de épocas; EarlyStopping pode parar antes
BATCH_SIZE      = 64       #
TAXA_APRENDIZADO = 0.001
DIR_SAIDAS      = os.path.join(BASE_DIR, "Saidas")


def carregar_dados():
    """
    Carrega o Fashion MNIST via Keras e normaliza os pixels para [0, 1].

    Retorna:
        (X_treino, y_treino, X_teste, y_teste): arrays NumPy normalizados.
        X_* tem formato (N, 28, 28) com valores float32 no intervalo [0, 1].
        y_* tem formato (N,) com inteiros de 0 a 9.
    """
    print("\n=== CARREGANDO FASHION MNIST ===")
    (X_treino, y_treino), (X_teste, y_teste) = keras.datasets.fashion_mnist.load_data()

    # Normalização: divide por 255 para colocar valores no intervalo [0, 1]
    X_treino = X_treino.astype(np.float32) / 255.0
    X_teste  = X_teste.astype(np.float32)  / 255.0

    print(f"Treino: {X_treino.shape[0]} amostras | Teste: {X_teste.shape[0]} amostras")
    print(f"Dimensão das imagens: {X_treino.shape[1]}x{X_treino.shape[2]} pixels\n")

    return X_treino, y_treino, X_teste, y_teste


def preparar_dados_brutos(X_treino, X_teste):
    """
    Adiciona o canal de cor (escala de cinza = 1 canal) exigido pelo Conv2D.

    Retorna:
        X_treino e X_teste com formato (N, 28, 28, 1).
    """
    return X_treino[..., np.newaxis], X_teste[..., np.newaxis]


def filtrar_binario(X, y):
    """
    Seleciona apenas as amostras das classes Camiseta=0 e Calças=1.
    Args:
        X: array com imagens ou descritores.
        y: array com rótulos inteiros (0–9).

    Retorna:
        (X_bin, y_bin): arrays filtrados com rótulos 0 para camiseta e 1 para calças
    """
    mascara = np.isin(y, CLASSES_BINARIA)
    X_bin = X[mascara]
    y_bin = y[mascara]

    # Remapeia: classe original 0 → índice 0, classe original 1 → índice 1
    # (neste caso já são 0 e 1, mas o remapeamento é explícito para generalidade)
    #MUDAR ISSO AQUI
    mapa = {cls: idx for idx, cls in enumerate(CLASSES_BINARIA)}
    y_bin = np.array([mapa[label] for label in y_bin])

    print(f"  Dados binários: {len(X_bin)} amostras "
          f"({np.sum(y_bin==0)} Camiseta / {np.sum(y_bin==1)} Calça)")
    return X_bin, y_bin


def executar_experimento(tarefa: str, modo_dados: str, X_treino, y_treino, X_teste, y_teste):
    """
    Executa um experimento completo: build -> fit -> teste.

    Args:
        tarefa:    'multiclasse' ou 'binaria'.
        modo_dados: 'bruto' ou 'hog_lbp'.
        X_treino, y_treino: dados de treinamento.
        X_teste,  y_teste:  dados de teste.
    """
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENTO: {tarefa.upper()} | {modo_dados.upper()}")
    print(f"{'#'*60}")

    cnn = CNN(
        tarefa=tarefa,
        modo_dados=modo_dados,
        epocas=EPOCAS,
        batch_size=BATCH_SIZE,
        taxa_aprendizado=TAXA_APRENDIZADO,
        diretorio_saida=DIR_SAIDAS
    )

    # Determina o shape/dim de entrada para o build do modelo
    if modo_dados == 'bruto':
        # Shape para Conv2D: (altura, largura, canais)
        input_arg = X_treino.shape[1:]
    else:
        # Dimensão do vetor HOG+LBP
        input_arg = X_treino.shape[1]

    cnn.build(input_arg)

    # Treina com 20% do treino usado como validação (para EarlyStopping)
    cnn.fit(X_treino, y_treino, X_val=None, y_val=None)
    # aqui usamos o split padrão do fit para simplicidade.
    # Para passar explicitamente, basta dividir X_treino antes e passar X_val/y_val.

    resultados = cnn.teste(X_teste, y_teste)

    print(f"\n  ✓ Acurácia final ({tarefa}, {modo_dados}): "
          f"{round(resultados['acuracia']*100, 2)}%")
    print(f"  ✓ Artefatos salvos em: {cnn.writer.pasta_atual}\n")


def main():
    # Reprodutibilidade
    tf.random.set_seed(42)
    np.random.seed(42)

    # --- Carregamento ---
    X_treino_raw, y_treino, X_teste_raw, y_teste = carregar_dados()

    # ============================================================
    # EXPERIMENTOS COM DADOS BRUTOS (Conv2D)
    # ============================================================
    X_treino_bruto, X_teste_bruto = preparar_dados_brutos(X_treino_raw, X_teste_raw)

    # Experimento 1: Multiclasse, dados brutos
    executar_experimento(
        tarefa='multiclasse',
        modo_dados='bruto',
        X_treino=X_treino_bruto, y_treino=y_treino,
        X_teste=X_teste_bruto,   y_teste=y_teste
    )

    # Experimento 2: Binária, dados brutos
    X_treino_bin_bruto, y_treino_bin = filtrar_binario(X_treino_bruto, y_treino)
    X_teste_bin_bruto,  y_teste_bin  = filtrar_binario(X_teste_bruto,  y_teste)

    executar_experimento(
        tarefa='binaria',
        modo_dados='bruto',
        X_treino=X_treino_bin_bruto, y_treino=y_treino_bin,
        X_teste=X_teste_bin_bruto,   y_teste=y_teste_bin
    )

    # ============================================================
    # EXPERIMENTOS COM DESCRITORES HOG+LBP (Dense)
    # ============================================================
    print("\n=== EXTRAINDO DESCRITORES HOG+LBP (pode demorar alguns minutos) ===")

    # Extrai sobre as imagens SEM o canal extra (shape N,28,28)
    X_treino_desc = extrair_hog_lbp(X_treino_raw)
    X_teste_desc  = extrair_hog_lbp(X_teste_raw)

    print(f"Dimensão do vetor HOG+LBP: {X_treino_desc.shape[1]}")

    # Experimento 3: Multiclasse, HOG+LBP
    executar_experimento(
        tarefa='multiclasse',
        modo_dados='hog_lbp',
        X_treino=X_treino_desc, y_treino=y_treino,
        X_teste=X_teste_desc,   y_teste=y_teste
    )

    # Experimento 4: Binária, HOG+LBP
    X_treino_bin_desc, y_treino_bin2 = filtrar_binario(X_treino_desc, y_treino)
    X_teste_bin_desc,  y_teste_bin2  = filtrar_binario(X_teste_desc,  y_teste)

    executar_experimento(
        tarefa='binaria',
        modo_dados='hog_lbp',
        X_treino=X_treino_bin_desc, y_treino=y_treino_bin2,
        X_teste=X_teste_bin_desc,   y_teste=y_teste_bin2
    )

    print("\n" + "="*60)
    print(" TODOS OS EXPERIMENTOS CONCLUÍDOS ".center(60))
    print("="*60)
    print(f" Artefatos salvos em: {DIR_SAIDAS}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
