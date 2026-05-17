"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054
"""

import pandas as pd
import numpy as np
from typing import Any

#Caracteres Completo
def ler_arquivo_csv(path_arquivo) -> Any:
    df = pd.read_csv(path_arquivo, header=None)
    return df

#Cada coluna n contém o conjunto de pesos que sai de todos os neurônios de entrada e vão para o
#neurônio n
def gera_matriz_pesos(num_entr, num_neur, ativacao='relu'):
    if ativacao == 'relu':
        std = np.sqrt(2.0 / num_entr)
        return np.random.randn(num_entr, num_neur) * std
    return np.random.uniform(-1, 1, size=(num_entr, num_neur))

def gera_matriz_bias(num_neur:int) -> Any:
    matriz=np.random.uniform(-1,1, size=(num_neur))
    return matriz

# inicio=gera_matriz_pesos(120, 10)
# resultado=vd_exemplo@inicio
# input_neuronios=resultado+gera_matriz_bias(10)