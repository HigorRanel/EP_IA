import numpy as np
from main.utils import ler_arquivo_csv


def main():
    x = ler_arquivo_csv('..\Entradas\CARACTERES COMPLETO\X.txt')
    y = ler_arquivo_csv('..\Entradas\CARACTERES COMPLETO\Y_letra.txt')

    x[120] = y[0]

if __name__ == '__main__':
    main()