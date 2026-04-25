import numpy as np
from utils import ler_arquivo_csv

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

    return dict_conversao

def main():
    x = ler_arquivo_csv('../Entradas/CARACTERES COMPLETO/X.txt')
    y = ler_arquivo_csv('../Entradas/CARACTERES COMPLETO/Y_letra.txt')

    colunas_letras = y[0]

    dict_conversao = criar_dict(colunas_letras)
    
    rotulos = np.array([dict_conversao[letra] for letra in colunas_letras])

    #X_np = x.values



if __name__ == '__main__':
    main()