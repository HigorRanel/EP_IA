import numpy as np
from utils import *
import ativacoes as atv


class MLP:

    def __init__(self, comprimento_entrada: int, comprimento_camada_oculta: int, comprimento_saida: int, epocas: int,
                 funcao_de_ativacao=atv.sigmoid, taxa_de_aprendizado=0.8):
        # Inicializando os hiper parâmetros do modelo
        self.comprimento_entrada = comprimento_entrada
        self.comprimento_camada_oculta = comprimento_camada_oculta
        self.comprimento_saida = comprimento_saida
        self.epocas = epocas

        # Inicializando as estruturas de dados dos pesos
        self.pesos_camada_oculta = gera_matriz_pesos(self.comprimento_entrada, self.comprimento_camada_oculta)
        self.bias_camada_oculta = gera_matriz_bias(self.comprimento_camada_oculta)
        self.comprimento_saida = gera_matriz_pesos(self.comprimento_saida, 120)
        self.bias_camada_saida = gera_matriz_bias(self.comprimento_saida)
        self.funcao_de_ativacao = funcao_de_ativacao

        # TODO: PENSAR EM COMO GUARDAR O HISTORICO DE ERROS?
        # TODO: GUARDAR INFOS DOS PESOS INICIAS - utils.py (ESCREVER EM UM TXT)

    def fit(self):
        for epoca in range(self.epocas):
            pass
            # TODO: CHECAR SE PESOS EPOCA ANTERIOR != PESOS EPOCA ATUAL

    def forward(self, entrada):
        # TODO: PENSAR EM COMO VAI SER PARA MAIS CAMADAS
        saida_camada_oculta = [0.0] * self.comprimento_camada_oculta

        # z_in = np.dot(entrada, self.pesos_camada_oculta) + self.bias_camada_oculta
        # saida_camada_oculta = self.funcao_de_ativacao(z_in)

        for i in range(self.comprimento_entrada):
            for j in range(self.comprimento_camada_oculta):
                saida_camada_oculta[i] += self.pesos_camada_oculta[j][i] * entrada[j]
            saida_camada_oculta[i] += self.bias_camada_oculta[i]
            # TODO: APLICAR FUNÇÃO DE ATIVAÇÃO NA SAÍDA CAMADA OCULTA

        for i in range(self.C):
            for j in range(self.comprimento_camada_oculta):
                saida_camada_oculta[i] += self.pesos_camada_oculta[j][i] * entrada[j]
            saida_camada_oculta[i] += self.bias_camada_oculta[i]
            # TODO: APLICAR FUNÇÃO DE ATIVAÇÃO NA SAÍDA CAMADA SAÍDA
        return saida_camada_oculta

    def backpropagate(self):
        pass

    def print_resultado(self):
        pass

    def check_fim_de_epoca(self):
        pass

    def set_funcao_de_ativacao(self, funcao):
        self.funcao_de_ativacao = funcao