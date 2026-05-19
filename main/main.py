"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054
"""

import numpy as np
from utils import ler_arquivo_csv
from mlp import *
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def criar_dict(y_col):
    """
    A função visa criar dicionário que mapeia cada letra para seu vetor
    Ex: {'A': [1,0,0, ...], 'B': [0,1,0, ...]}
    """
    ordem_alfabetica = list(y_col.unique())
    ordem_alfabetica.sort()

    dict_conversao = {}
    for i in list(ordem_alfabetica):
        lista = [0] * 26
        indice = list(ordem_alfabetica).index(i)
        lista[indice] = 1
        dict_conversao[i] = lista

    return ordem_alfabetica, dict_conversao

def holdout_estratificado(x, valor_esperado_df, rotulos, colunas_letras, test_size=0.3, seed=42):
    np.random.seed(seed)

    total = len(x)
    n_teste_total = int(total * test_size)   # ~398
    indices_fixos_teste = list(range(total - 130, total))
    indices_restantes = list(range(total - 130))  # 1196

    # Quantas amostras ainda precisam ir para o teste além dos 130 fixos
    n_teste_extra = n_teste_total - 130  # ~268

    indices_treino = []
    indices_teste_extra = []

    colunas_letras_restantes = colunas_letras.iloc[indices_restantes]

    for letra in sorted(colunas_letras_restantes.unique()):
        indices_letra = np.where(colunas_letras_restantes == letra)[0]
        np.random.shuffle(indices_letra)

        # Proporção do extra estratificada por classe
        n_extra_letra = max(1, int(len(indices_letra) * (n_teste_extra / len(indices_restantes))))

        indices_teste_extra.extend(indices_restantes[i] for i in indices_letra[:n_extra_letra])
        indices_treino.extend(indices_restantes[i] for i in indices_letra[n_extra_letra:])

    indices_teste = indices_fixos_teste + indices_teste_extra

    treino_x = x.iloc[indices_treino, :]
    treino_y = valor_esperado_df.iloc[indices_treino, :]
    rotulos_treino = rotulos[indices_treino]

    teste_x = x.iloc[indices_teste, :]
    teste_y = valor_esperado_df.iloc[indices_teste, :]
    rotulos_teste = rotulos[indices_teste]

    print(f"\n=== DIVISÃO HOLDOUT ESTRATIFICADO (test_size={test_size}, seed={seed}) ===")
    print(f"Total:  {total} amostras")
    print(f"Treino: {len(treino_x)} amostras ({round(len(treino_x)/total*100, 1)}%)")
    print(f"Teste:  {len(teste_x)} amostras  ({round(len(teste_x)/total*100, 1)}%)")
    print(f" 130 fixos (finais) + {len(indices_teste_extra)} via estratificado")
    print("=" * 55 + "\n")

    return treino_x, treino_y, rotulos_treino, teste_x, teste_y, rotulos_teste

def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ENTRADAS = os.path.join(BASE_DIR, 'Entradas', 'CARACTERES COMPLETO')

    x = ler_arquivo_csv(os.path.join(ENTRADAS, 'X.txt'))
    y = ler_arquivo_csv(os.path.join(ENTRADAS, 'Y_letra.txt'))
    # mlp = MLP(
    #     120,
    #     90,
    #     26,
    #     epocas=100,
    #     taxa_de_aprendizado=0.7,
    #     limiar_erro=0.02
    # )

    mlp = MLP(
        120,
        90,
        26,
        epocas=100,
        taxa_de_aprendizado=0.3,
        limiar_erro=0.0055
    )

    # mlp = MLP(
    #     120,
    #     150,
    #     26,
    #     epocas=200,
    #     taxa_de_aprendizado=0.5,
    #     limiar_erro=0.02
    # )
    colunas_letras = y[0]
    valor_esperado_df = y[[0]]

    letras, dict_conversao = criar_dict(colunas_letras)
    
    rotulos = np.array([dict_conversao[letra] for letra in colunas_letras])

    #X_np = x.values
    # print(x.iloc[:, 120])
    x=x.drop(columns={120})
    # print(x.iloc[:, 119])

    treino_x, treino_y, rotulos_treino, teste_x, teste_y, rotulos_teste = holdout_estratificado(x, valor_esperado_df,
                                                                                           rotulos,
                                                                                           colunas_letras,
                                                                                           test_size=0.3,
                                                                                           seed=42)
    # UTILIZANDO APENAS OS 130 DADOS FINAIS NO TREINAMENTO
    # treino_x = x.iloc[:-130]
    # treino_y = valor_esperado_df.iloc[:-130]
    # rotulos_treino = rotulos[:-130]
    #
    # teste_x = x.iloc[-130:]
    # teste_y = valor_esperado_df.iloc[-130:]
    # rotulos_teste = rotulos[-130:]
    #####################################################

    mlp.fit(treino_x, rotulos_treino)
    resultados = mlp.teste(teste_x, rotulos_teste, letras, teste_y)

    # Gera e exibe a matriz de confusão
    mlp.matriz_confusao(resultados, letras)


if __name__ == '__main__':
    main()