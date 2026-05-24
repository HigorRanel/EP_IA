"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

Módulo que implementa as funções de ativação
"""

import numpy as np

def sigmoid(inpt):
    """
    Calcula a função de ativação sigmoid

    Recebe como parâmetros:
    1) inpt: valor de entrada (ou array NumPy)

    Retorna: 1 / (1 + e^(-inpt))
    """
    return 1/(1+np.exp(-inpt))

def derivada_sigmoid(inpt):
    """
    Calcula a derivada da função sigmoid, usada no backpropagation

    Recebe como parâmetros:
    1) inpt: valor de entrada (ou array NumPy)

    Retorna: sigmoid(inpt) * (1 - sigmoid(inpt))
    """
    # A derivida da sigmoid é matematicamente definida como:
    return sigmoid(inpt) * (1 - sigmoid(inpt))