import pandas as pd
import numpy as np

#MLP com UMA camada escondida
#Só é obrigatório usar o caracteres completo. Os outros são para teste
#Quantos neurônios na camada intermediária? Ninguém sabe ao certo. É um problema de otimização

#Caracteres Completo
X = pd.read_csv(r'C:\Users\luizv\PyCharmMiscProject\EP_IA\Entradas\CARACTERES COMPLETO\X.txt', header=None)
Y = pd.read_csv(r'C:\Users\luizv\PyCharmMiscProject\EP_IA\Entradas\CARACTERES COMPLETO\Y_letra.txt', header=None)
#print(X[120].value_counts())
X[120]=Y[0]
#print(X[120].value_counts())

#Funções de ativação
def step_t(inpt:float, num:float)->float:
    if inpt >= num:
        return 1
    else:
        return 0

def lin_part(inpt:float)->float:
    if inpt<=-1/2:
        return -1
    elif inpt<1/2:
        return inpt
    else:
        return 1

def sigmoid(inpt:float)->float:
    return 1/(1+np.exp(-inpt))

def relu(x):
    return np.maximum(0, x)

#Para construir a rede neural de vardade, vale muito a pena usar orientação a objetos. Como eu não
#fazer isso, eu vou só criar algumas estruturas como funções e dps vcs fazem oq vcs quiserem
#Vou construir matrizes para os pesos. Tem que transformar a linha em um vetor "deitado"

vd_exemplo=np.array(X.iloc[0][0:120]).reshape(1, -1)
print(vd_exemplo)

def matriz_inicio(num_entr:int, num_neur:int):
    matriz=np.random.randn(num_entr,num_neur)
    return matriz

def bias(num_neur:int):
    matriz=np.random.randn(1, num_neur)
    print(matriz)
    return matriz
inicio=matriz_inicio(120, 10)

resultado=vd_exemplo@inicio
input_neuronios=resultado+bias(10)