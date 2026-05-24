"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

Módulo de funções utilitárias para leitura de arquivos e geração de matriz de pesos
"""

import pandas as pd
import numpy as np

# Caracteres Completo
def ler_arquivo_csv(path_arquivo):
    """
    Lê um arquivo CSV e retorna o dataframe correspondente
    """
    df = pd.read_csv(path_arquivo, header=None)
    return df

# A inicialização dos pesos utiliza a proposta de Glorot e Bengio, tal método é chamado de
# inicialização de Xavier. Para funções de ativação simétricas como a sigmoid, surge a necessidade de
# manter a variância do sinal constante entre as camadas durante o feedforward e o backpropagate,
# de modo a evitar os problemas de gradiente explodindo
# Os pesos são amostrados no seguinte intervalo U[-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out))]

def gera_matriz_pesos(num_entr, num_neur):
    """
    Gera a matriz de pesos de uma camada com inicialização de Xavier (explicado acima)

    Recebe como parâmetros:
    1) num_entr: nº de entradas da camada (n_in)
    2) num_neur: nº de neurônios da camada (n_out)

    Retorna: Matriz NumPy de dimensões num_entr x num_neur) com pesos inicializados em um intervalo:
        U[-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
    """
    limite = np.sqrt(6.0 / (num_entr + num_neur))
    return np.random.uniform(-limite, limite,
                             size=(num_entr, num_neur))

def gera_matriz_bias(num_neur):
    """
    Gera o vetor de bias de uma camada, inicializado com zeros

    Recebe como parâmetros:
    1) num_neur: nº de neurônios da camada

    Retorna: Vetor NumPy de zeros com tamanho num_neur
    """
    return np.zeros(num_neur)

