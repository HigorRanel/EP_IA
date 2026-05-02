import numpy as np

from ativacoes import derivada_sigmoid
from utils import *
import ativacoes as atv
import loggers.logger as logger

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
        self.W = gera_matriz_pesos(self.comprimento_entrada, self.comprimento_oculta)
        self.w0 = gera_matriz_bias(self.comprimento_oculta)
        self.B = gera_matriz_pesos(self.comprimento_oculta, self.comprimento_saida)
        self.b0 = gera_matriz_bias(self.comprimento_saida)

        self.erros = []
        self.y = []
        self.y_in = []
        self.z = []
        self.z_in = []

        # Instancia o Logger e exibe as configurações iniciais
        self.logger = logger.Logger()
        nome_ativacao = self.funcao_de_ativacao.__name__ if hasattr(self.funcao_de_ativacao,
                                                                    '__name__') else "Personalizada"
        self.logger.log_configuracoes_iniciais(
            self.comprimento_entrada, self.comprimento_oculta, self.comprimento_saida,
            self.taxa_de_aprendizado, nome_ativacao
        )

    def fit(self, dados, rotulos):
        for epoca in range(self.epocas):
            self.logger.log_inicio_epoca(epoca, self.epocas)

            for i in range(dados.shape[0]):
                self.logger.log_iteracao_dado(i)
                dado = dados.iloc[i]

                self.logger.log_entrada(dado)
                self.forward(dado)

                resp = rotulos[i]
                self.backpropagate(dado, resp, self.z_in, self.z, self.y_in, self.y)

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

        # Camada oculta
        for j in range(self.comprimento_oculta):
            self.z_in[j] = self.w0[j]
            for i in range(self.comprimento_entrada):
                self.z_in[j] += entrada[i] * self.W[i][j]
            self.z[j] = self.funcao_de_ativacao(self.z_in[j])

        self.logger.log_camada_oculta(self.W, self.z_in, self.z)

        # Camada de saída
        self.y_in = [0.0] * self.comprimento_saida
        self.y = [0.0] * self.comprimento_saida

        for k in range(self.comprimento_saida):
            self.y_in[k] = self.b0[k]
            for j in range(self.comprimento_oculta):
                self.y_in[k] += self.z[j] * self.B[j][k]
            self.y[k] = self.funcao_de_ativacao(self.y_in[k])

        self.logger.log_camada_saida(self.B, self.y_in, self.y)

    def teste(self, dados, rotulos, letras, valor_esperado):
        count = 0
        for i in range(dados.shape[0]):
            self.logger.log_iteracao_teste(i)

            self.forward(dados.iloc[i])
            saida = np.array(self.y)
            saida_arredondada = list(np.round(saida, 2))
            indice_letra = saida_arredondada.index(max(saida_arredondada))

            prev = letras[indice_letra]
            esp = valor_esperado.iat[i, 0]

            if (prev.casefold() == esp.casefold()):
                count += 1

            resp = rotulos[i]
            erro = np.array(self.y) - np.array(resp)

            self.logger.log_resultado_teste(prev, esp, saida_arredondada, sum(erro))

        print(f'-----------------------------------ACURACIA-----------------------------------')
        print(f'Acurácia= {count}/{dados.shape[0]} = {count / dados.shape[0]}')

    def backpropagate(self, x, t, z_in, z, y_in, y):
        # informação de erro da camada de saída
        deltaMaior_k = [0.0] * self.comprimento_saida
        delta_b_jk = [[0.0] * self.comprimento_saida for _ in range(self.comprimento_oculta)]
        delta_b0j = [0.0] * self.comprimento_saida

        for k in range(self.comprimento_saida):
            deltaMaior_k[k] = (t[k] - y[k]) * derivada_sigmoid(y_in[k])
            delta_b0j[k] = self.taxa_de_aprendizado * deltaMaior_k[k]

            for j in range(self.comprimento_oculta):
                delta_b_jk[j][k] = self.taxa_de_aprendizado * deltaMaior_k[k] * z[j]

        # informação de erro da camada oculta
        deltaMaior_in_j = [0.0] * self.comprimento_oculta
        deltaMaior_j = [0.0] * self.comprimento_oculta
        delta_W = [[0.0] * self.comprimento_oculta for _ in range(self.comprimento_entrada)]
        delta_w0 = [0.0] * self.comprimento_oculta

        for j in range(self.comprimento_oculta):
            for k in range(self.comprimento_saida):
                deltaMaior_in_j[j] += deltaMaior_k[k] * self.B[j][k]

            deltaMaior_j[j] = deltaMaior_in_j[j] * derivada_sigmoid(z_in[j])
            delta_w0[j] = self.taxa_de_aprendizado * deltaMaior_j[j]

            for i in range(self.comprimento_entrada):
                delta_W[i][j] = self.taxa_de_aprendizado * deltaMaior_j[j] * x[i]

        # atualização de pesos
        for k in range(self.comprimento_saida):
            self.b0[k] += delta_b0j[k]
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
        erro_total = 0.5 * erro

        self.logger.log_backprop_erros(y, t, erro_total)
        self.logger.log_erro_saida(deltaMaior_k, delta_b_jk, delta_b0j)
        self.logger.log_erro_oculta(deltaMaior_in_j, deltaMaior_j, delta_W, delta_w0)
        self.logger.log_atualizacao_pesos(self.B, self.b0, self.W, self.w0)

        return erro_total

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
            elif isinstance(dado, list):
                filehandler.write(dado[0])
                for linha in dado[1:]:
                    filehandler.write(" | ".join(str(item) for item in linha))

    def check_limiar_de_erro(self):
        pass

    def set_funcao_de_ativacao(self, funcao):
        self.funcao_de_ativacao = funcao