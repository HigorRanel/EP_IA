import pandas as pd
import numpy as np
from typing import Any

#MLP com UMA camada escondida
#Só é obrigatório usar o caracteres completo. Os outros são para teste
#Quantos neurônios na camada intermediária? Ninguém sabe ao certo. É um problema de otimização

#Caracteres Completo
def ler_arquivo_csv(path_arquivo) -> Any:
    df = pd.read_csv(path_arquivo, header=None)
    return df

#Para construir a rede neural de vardade, vale muito a pena usar orientação a objetos. Como eu não
#fazer isso, eu vou só criar algumas estruturas como funções e dps vcs fazem oq vcs quiserem
#Vou construir matrizes para os pesos. Tem que transformar a linha em um vetor "deitado"

# vd_exemplo=np.array(X.iloc[0][0:120]).reshape(1, -1)
# print(vd_exemplo)

#Cada coluna n contém o conjunto de pesos que sai de todos os neurônios de entrada e vão para o
#neurônio n
def gera_matriz_pesos(num_entr:int, num_neur:int) -> Any:
    matriz=np.random.uniform(-1,1, size=(num_entr, num_neur))
    return matriz

def gera_matriz_bias(num_neur:int) -> Any:
    matriz=np.random.uniform(-1,1, size=(num_neur))
    print(matriz)
    return matriz

# inicio=gera_matriz_pesos(120, 10)
# resultado=vd_exemplo@inicio
# input_neuronios=resultado+gera_matriz_bias(10)