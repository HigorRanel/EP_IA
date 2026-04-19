import numpy as np

from main.ativacoes import derivada_sigmoid
from utils import *
import ativacoes as atv

"""
Nomes e Nº USP:

"""


class MLP:
    """

    Implementação da Rede Neural Mutilayer Perceptron (MLP)
    com uma camada oculta, treinada com Backpropagation
    (Gradiente Descendente)
    
    Pesos camada oculta: W[i][j]: matriz(comprimento_entrada X comprimento_oculta)
    W_ij

    Bias camada oculta: v0[j]: vetor(comprimento_oculta)
    w_0j

    Pesos camada saída: B[j][k]: matriz (comprimento_oculta x comprimento_saida)
    B_jk

    Bias camada saída: b0[k] vetor(comprimento_saida)
    b_0k

    """

    def __init__(self, comprimento_entrada: int, comprimento_oculta: int, comprimento_saida: int, epocas: int,
                 funcao_de_ativacao=atv.sigmoid, taxa_de_aprendizado: float = 0.8):

        # Inicializando os hiperparâmetros do modelo
        self.comprimento_entrada = comprimento_entrada
        self.comprimento_oculta = comprimento_oculta
        self.comprimento_saida = comprimento_saida
        self.epocas = epocas
        self.funcao_de_ativacao = funcao_de_ativacao
        self.taxa_de_aprendizado = taxa_de_aprendizado

        # Inicializando as estruturas de dados dos pesos

        # W[i][j]: aqui temos os pesos entre a entrada i e o neurônio j
        self.W = gera_matriz_pesos(self.comprimento_entrada, self.comprimento_oculta)

        # w0[j]: bias do neurônio oculto j
        self.w0 = gera_matriz_bias(self.comprimento_oculta)

        # B[j][k]: peso entre neurônio oculto j e saída k
        self.B = gera_matriz_pesos(self.comprimento_oculta, self.comprimento_saida)

        # b0[k]: bias do neurônio de saída k
        self.b0 = gera_matriz_bias(self.comprimento_saida)

        # Salva os erros por época
        self.erros = []

        # Salva pesos iniciais
        # TODO: GUARDAR INFOS DOS PESOS INICIAS - utils.py (ESCREVER EM UM TXT)
        #self._salvar_pesos("pesos_iniciais.txt", self.V, self.v0, self.W, self.w0)



    def fit(self):
        for epoca in range(self.epocas):
            pass
            # TODO: CHECAR SE PESOS EPOCA ANTERIOR != PESOS EPOCA ATUAL E OUTROS TIPOS DE PARADAS

    def forward(self, entrada):
        """
        camada oculta:
        z_in_j = w_0j + somatório de 1 até comp camada entrada (entrada_i * w_ij)
        z_j = f (z_in_j)

        camada de saída:
        y_in_k = b_0k + somatório de 1 até comp camada oculta (z_j * bw_jk)
        y_k = f(y_in_k)

        """

        z_in = [0.0] * self.comprimento_oculta
        z = [0.0] * self.comprimento_oculta

        # Camada oculta:

        # Aqui iremos aplicar o z_in_j sendo o somatório de cada entrada multiplicado pelo
        # respectivo peso e no final soma-se esse valor ao bias do neurôrio j da camada
        # oculta
        for j in range(self.comprimento_oculta):
            z_in[j] = self.w0[j]
            for i in range(self.comprimento_entrada):
                z_in[j] += entrada[i] * self.W[i][j]
            # aplicamos a função de ativação em z_in_j
            z[j] = self.funcao_de_ativacao(z_in[j])

        # Camada de saída

        y_in = [0.0] * self.comprimento_saida
        y = [0.0] * self.comprimento_saida

        for k in range(self.comprimento_saida):
            #  y_in_k = b_0k + somatório de 1 até comp camada oculta (z_j * bw_jk)
            y_in[k] = self.b0[k]
            for j in range(self.comprimento_oculta):
                y_in[j] += z[j] * self.B[j][k]
            # y_k = f(y_in_k)
            y[k] = self.funcao_de_ativacao(y_in[k])

        # TODO: CRIAR FUNÇÃO DE FOR PARA GENERALIZAR OS FOR ACIMA


    def backpropagate(self, x, t, z_in, z, y_in, y):
        """

            informação de erro da camada de saída:
            deltaMaior_k = (target_k - y_k) * f'(y_in_k)
            delta_b_jk = alpha(taxa de apredizado) * deltaMaior_k * z_j
            delta_b_0k = alpha * deltaMaior_k

            retropropagação para camada oculta:
            in_j = somatorio de k = 1 até comprimento saida (deltaMaior_k * b_jk )
            deltaMaior_j = deltaMaior_in_j * f'(z_in_j)
            delta_w_ij = αlpha * deltaMaior_j * x_i
            delta_w_0j = αlpha * deltaMaior_j

            atualização de pesos:
            b_jk(new) = b_jk(old) + delta_b_jk
            w_ij(new) = w_ij(old) + delta_w_ij

        """

        # informação de erro da camada de saída
        deltaMaior_k = [0.0] * self.comprimento_saida

        # cria matriz para guardar o delta de cada erro (cada linha é representa um neurônio da camada oculta)
        delta_b_jk = [[0.0] * self.comprimento_saida for _ in range(self.comprimento_oculta)]

        delta_b0j = [0.0] * self.comprimento_saida

        for k in range(self.comprimento_saida):
            # deltaMaior = (target_k - y_k)*f'(i_in_k)
            deltaMaior_k[k] = (t[k] - y[k]) * derivada_sigmoid(y[k])

            # delta b_0k = alpha * deltaMaior_k
            delta_b0j = self.taxa_de_aprendizado * deltaMaior_k[k]

            # delta_b_0k = alpha * delta_maior_k * z_j
            for j in range(self.comprimento_oculta):
                delta_b_jk[k][j] = self.taxa_de_aprendizado * deltaMaior_k[k] * z[j]

        # CONTINUAR ...

    def print_resultado(self):
        pass

    def check_fim_de_epoca(self):
        pass

    def set_funcao_de_ativacao(self, funcao):
        self.funcao_de_ativacao = funcao