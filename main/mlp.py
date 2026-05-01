import numpy as np

from ativacoes import derivada_sigmoid
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

    Bias camada oculta: w0[j]: vetor(comprimento_oculta)
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

        #
        self.y = [] #saída dos neurônios da camada de saída

        self.y_in = [] #entrada dos neurônios de saída

        self.z = [] #saída dos neurônios da camada oculta

        self.z_in =[] #entrada dos neurônios da camada oculta


        # Salva pesos iniciais
        # TODO: GUARDAR INFOS DOS PESOS INICIAS - utils.py (ESCREVER EM UM TXT)
        # self._salvar_pesos("pesos_iniciais.txt", self.V, self.v0, self.W, self.w0)



    def fit(self, dados, rotulos):
        for epoca in range(self.epocas):
            for i in range(dados.shape[0]):
                dado = dados.iloc[i]
                self.forward(dado)
                resp = rotulos[i]
                #erro = np.array(self.y)-np.array(resp)
                self.backpropagate(dado, resp, self.z_in, self.z, self.y_in, self.y)
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

        self.z_in = [0.0] * self.comprimento_oculta
        self.z = [0.0] * self.comprimento_oculta

        # Camada oculta:

        # Aqui iremos aplicar o z_in_j sendo o somatório de cada entrada multiplicado pelo
        # respectivo peso e no final soma-se esse valor ao bias do neurôrio j da camada
        # oculta
        for j in range(self.comprimento_oculta):
            self.z_in[j] = self.w0[j]
            for i in range(self.comprimento_entrada):
                self.z_in[j] += entrada[i] * self.W[i][j]
            # aplicamos a função de ativação em z_in_j
            self.z[j] = self.funcao_de_ativacao(self.z_in[j])

        # Camada de saída

        self.y_in = [0.0] * self.comprimento_saida
        self.y = [0.0] * self.comprimento_saida

        for k in range(self.comprimento_saida):
            #  y_in_k = b_0k + somatório de 1 até comp camada oculta (z_j * bw_jk)
            self.y_in[k] = self.b0[k]
            for j in range(self.comprimento_oculta):
                self.y_in[k] += self.z[j] * self.B[j][k]
            # y_k = f(y_in_k)
            self.y[k] = self.funcao_de_ativacao(self.y_in[k])

        # TODO: CRIAR FUNÇÃO DE FOR PARA GENERALIZAR OS FOR ACIMA

    def teste(self, dados, rotulos):
        for i in range(dados.shape[0]):
            print(f'-----------------------------------{i}-----------------------------------')
            self.forward(dados.iloc[i])
            saida=np.array(self.y)
            saida=np.round(saida, 2)
            print(f'Saída:    {saida}')
            resp = rotulos[i]
            erro = np.array(self.y)-np.array(resp)
            print(f'Resposta: {resp}')
            print(f'Erro:     {sum(erro)}')




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
            deltaMaior_k[k] = (t[k] - y[k]) * derivada_sigmoid(y_in[k])

            # delta b_0k = alpha * deltaMaior_k
            delta_b0j = self.taxa_de_aprendizado * deltaMaior_k[k]

            # delta_b_0k = alpha * delta_maior_k * z_j
            for j in range(self.comprimento_oculta):
                delta_b_jk[j][k] = self.taxa_de_aprendizado * deltaMaior_k[k] * z[j]

        # informação de erro da camada oculta
        deltaMaior_in_j = [0.0] * self.comprimento_oculta
        deltaMaior_j = [0.0] * self.comprimento_oculta
        delta_W = [[0.0] * self.comprimento_oculta for _ in range(self.comprimento_entrada)]
        delta_w0 = [0.0] * self.comprimento_oculta

        for j in range(self.comprimento_oculta):
            # deltaMaior_in_j = somatório de k = 1 até m (deltaMaior_k * b_jk)
            for k in range(self.comprimento_saida):
                deltaMaior_in_j[j] += deltaMaior_k[k] * self.B[j][k]

            # deltaMaior_j = deltaMaior_in_j * f'(z_in_j)
            deltaMaior_j[j] = deltaMaior_in_j[j] * derivada_sigmoid(z_in[j])

            # delta_w_0j = alpha * deltaMaior_j
            delta_w0[j] = self.taxa_de_aprendizado * deltaMaior_j[j]

            # delta_ij = alpha * deltaMaior_j * x_i
            for i in range(self.comprimento_entrada):
                delta_W[i][j] = self.taxa_de_aprendizado * deltaMaior_j[j] * x[i]

        # atualização de pesos
        for k in range(self.comprimento_saida):
            self.b0[k] += delta_b0j
            for j in range(self.comprimento_oculta):
                self.B[j][k] += delta_b_jk[j][k]

        for j in range(self.comprimento_oculta):
            self.w0[j] += delta_w0[j]
            for i in range(self.comprimento_entrada):
                self.W[i][j] += delta_W[i][j]

        # erro quadrático: E(0) = 0.5 * sum_k(e_^2)
        erro = 0.0
        for k in range(self.comprimento_saida):
            erro += (t[k] - y[k]) ** 2
        return 0.5 * erro



    def print_console(self, dado):
        if isinstance(dado, dict):
            for chave, valor in dado.items():
                print(f"{chave}: {valor}")

        elif isinstance(dado, list):
            for linha in dado:
                print(" | ".join(str(item) for item in linha))

        elif isinstance(dado, str):
            print(dado)
    
    def print_arquivo(self, nome_arquivo, dado):
        with open(nome_arquivo, 'w') as filehandler:
            if isinstance(dado, dict):
                for chave, valor in dado.items():
                    filehandler.write(f"{chave}: {valor}")

            # a primeira linha seria um título para a tabela
            elif isinstance(dado, list):
                filehandler.write(dado[0])
                for linha in dado[1:]:
                    filehandler.write(" | ".join(str(item) for item in linha))

    def check_limiar_de_erro(self):
        pass

    def set_funcao_de_ativacao(self, funcao):
        self.funcao_de_ativacao = funcao