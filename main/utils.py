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

# A inicialização dos pesos segue a proposta de Glorot e Bengio (2010), conhecida como
# inicialização de Xavier. Para funções de ativação simétricas como a sigmoid, é importante
# manter a variância do sinal constante entre as camadas durante o forward e o backpropagate,
# de forma a evitar os problemas de gradiente explodindo.
# Os pesos são amostrados de U[-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out))].
def gera_matriz_pesos(num_entr, num_neur):
    limite = np.sqrt(6.0 / (num_entr + num_neur))
    return np.random.uniform(-limite, limite, size=(num_entr, num_neur))

def gera_matriz_bias(num_neur: int) -> Any:
    return np.zeros(num_neur)

