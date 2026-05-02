import numpy as np
from utils import ler_arquivo_csv
from mlp import *

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

def main():
    x = ler_arquivo_csv('../Entradas/CARACTERES COMPLETO/X.txt')
    y = ler_arquivo_csv('../Entradas/CARACTERES COMPLETO/Y_letra.txt')
    mlp=MLP(120, 100, 26, 5)

    colunas_letras = y[0]
    valor_esperado_df = y[[0]]

    letras, dict_conversao = criar_dict(colunas_letras)
    
    rotulos = np.array([dict_conversao[letra] for letra in colunas_letras])

    #X_np = x.values
    # print(x.iloc[:, 120])
    x=x.drop(columns={120})
    # print(x.iloc[:, 119])

    treino_percent=int(0.6*x.shape[0])
    teste_percent=int(0.2*x.shape[0])

    treino_x=x.iloc[0:treino_percent,:]
    treino_y=valor_esperado_df.iloc[0:treino_percent,:]
    rotulos_treino=rotulos[0:treino_percent]

    teste_x=x.iloc[treino_percent:treino_percent+teste_percent, :]
    teste_y=valor_esperado_df.iloc[treino_percent:treino_percent+teste_percent, :]
    rotulos_teste=rotulos[treino_percent:treino_percent+teste_percent]

    print(len(rotulos_treino), len(treino_x))

    mlp.fit(treino_x, rotulos_treino)
    mlp.teste(teste_x, rotulos_teste, letras, teste_y)

    # print(treino_x)
    # print(treino_y)
    # print(teste_x)
    # print(teste_y)

if __name__ == '__main__':
    main()