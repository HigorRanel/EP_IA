"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ativacoes import derivada_sigmoid
from utils import *
import ativacoes as atv
from loggers.logger import Logger
from loggers.writer import Writer
import plots


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
                 funcao_de_ativacao=atv.sigmoid, taxa_de_aprendizado: float = 0.8,
                 verbose: bool = False, paciencia: int = 10):

        # Inicializando os hiperparâmetros do modelo
        self.comprimento_entrada = comprimento_entrada
        self.comprimento_oculta = comprimento_oculta
        self.comprimento_saida = comprimento_saida
        self.epocas = epocas
        self.funcao_de_ativacao = funcao_de_ativacao
        self.taxa_de_aprendizado = taxa_de_aprendizado
        self.verbose = verbose
        # Paciência da parada antecipada: nº de épocas sem melhora no erro de
        # validação toleradas antes de interromper o treinamento.
        self.paciencia = paciencia

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
        self.erros = []        # EQM de treino por época
        self.erros_val = []    # EQM de validação por época (vazio se não houver validação)

        #
        self.y = []  # saída dos neurônios da camada de saída

        self.y_in = []  # entrada dos neurônios de saída

        self.z = []  # saída dos neurônios da camada oculta

        self.z_in = []  # entrada dos neurônios da camada oculta

        # Inicializa logger e writer
        self.logger = Logger(verbose=self.verbose)
        self.writer = Writer()

        # Loga e salva configurações iniciais
        self.logger.log_configuracoes_iniciais(
            comprimento_entrada, comprimento_oculta, comprimento_saida,
            taxa_de_aprendizado, funcao_de_ativacao.__name__
        )
        self.writer.write_hiperparametros(
            comprimento_entrada, comprimento_oculta, comprimento_saida,
            taxa_de_aprendizado, funcao_de_ativacao.__name__,
            epocas
        )

        self.erros_iteracao = []  # lista de {epoca, iteracao, erro}
        # self._salvar_pesos("pesos_iniciais.txt", self.V, self.v0, self.W, self.w0)
        self.writer.write_pesos(self.W, self.w0, self.B, self.b0, etapa="iniciais")

    def fit(self, dados, rotulos, val_dados=None, val_rotulos=None):
        """
        Treina a rede com Backpropagation (Gradiente Descendente).

        Se val_dados/val_rotulos forem fornecidos, ao final de cada época o
        EQM de validação é calculado e a PARADA ANTECIPADA é aplicada: o
        treinamento é interrompido quando o erro de validação não melhora por
        `self.paciencia` épocas consecutivas. Não há restauração de pesos — os
        pesos finais são os da última época treinada.

        Args:
            dados, rotulos:         conjunto de TREINO.
            val_dados, val_rotulos: conjunto de VALIDAÇÃO (opcional).
        """
        usar_validacao = val_dados is not None and val_rotulos is not None

        # Estado da parada antecipada
        melhor_erro_val = float('inf')
        epocas_sem_melhora = 0

        for epoca in range(self.epocas):

            # O embaralhamento por época é necessário pois o holdout estratificado agrupa as amostras
            # por classe. Sem isso, a rede veria todas as variações de "A", depois todas de "B", e assim
            # por diante, de forma que os pesos sejam ajustados repetidamente para uma única letra
            # por vez, prejudicando o que foi aprendido sobre as letras anteriores. Ao embaralhar,
            # garantimos que a rede aprenda todas as letras de forma intercalada a cada época.
            indices = np.random.permutation(dados.shape[0])
            dados_epoca = dados.iloc[indices].reset_index(drop=True)
            rotulos_epoca = rotulos[indices]

            self.logger.log_inicio_epoca(epoca, self.epocas)
            erros_epoca = []
            n_amostras = dados_epoca.shape[0]
            for i in range(n_amostras):
                self.logger.log_iteracao_dado(i)
                dado = dados_epoca.iloc[i]
                self.forward(dado)
                resp = rotulos_epoca[i]
                erro = self.backpropagate(dado, resp, self.z_in, self.z, self.y_in, self.y)
                erros_epoca.append(erro)
                self.erros_iteracao.append({
                    'epoca': epoca+1,
                    'iteracao': i+1,
                    'erro': erro
                })
                # Barra de progresso estilo Keras (silenciada se verbose=True)
                erro_parcial = sum(erros_epoca) / len(erros_epoca)
                self.logger.barra_progresso_epoca(
                    epoca, self.epocas, i + 1, n_amostras, erro_medio=erro_parcial
                )

            # EQM de treino da época
            erro_medio = sum(erros_epoca) / len(erros_epoca) if erros_epoca else 0.0
            self.erros.append(erro_medio)

            # ----- PARADA ANTECIPADA POR VALIDAÇÃO -----
            if usar_validacao:
                erro_val = self._eqm_conjunto(val_dados, val_rotulos)
                self.erros_val.append(erro_val)
                self.logger.log_erro_validacao(epoca + 1, erro_medio, erro_val)

                # Houve melhora no erro de validação?
                if erro_val < melhor_erro_val:
                    melhor_erro_val = erro_val
                    epocas_sem_melhora = 0
                else:
                    epocas_sem_melhora += 1

                # Interrompe se a paciência foi excedida
                if epocas_sem_melhora >= self.paciencia:
                    self.logger.log_parada_antecipada_val(epoca + 1, erro_val, self.paciencia)
                    break

        # Salva pesos finais e erros ao fim do treinamento
        self.writer.write_pesos(self.W, self.w0, self.B, self.b0, etapa="finais")
        self.writer.write_erros(self.erros, self.erros_iteracao, self.erros_val)

        # Gráfico de comportamento do erro (treino vs. validação) por época
        caminho_grafico = self.writer._obter_caminho("8_curva_erro.png")
        plots.plotar_curva_erro(
            self.erros,
            self.erros_val if self.erros_val else None,
            caminho_saida=caminho_grafico
        )

    def _eqm_conjunto(self, dados, rotulos):
        """
        Calcula o Erro Quadrático Médio (EQM) sobre um conjunto inteiro,
        executando apenas o forward (sem ajuste de pesos). Usado para medir
        o erro de validação ao final de cada época.

        Convenção de Haykin (slides 55-56):
            E_av = (1/N) * Σ_amostras E(n),  com  E(n) = (1/2) * Σ_k (t_k - y_k)^2

        onde N é o nº de amostras e k percorre os neurônios da camada de saída.
        """
        m = self.comprimento_saida
        soma = 0.0
        n = dados.shape[0]
        for i in range(n):
            self.forward(dados.iloc[i])
            t = np.array(rotulos[i])
            y = np.array(self.y)
            # E(n) = (1/2) * Σ_k (t_k - y_k)^2
            soma += 0.5 * np.sum((t[:m] - y[:m]) ** 2)
        return soma / n if n > 0 else 0.0  # E_av = (1/N) Σ_n E(n)

    def forward(self, entrada):
        """
        camada oculta:
        z_in_j = w_0j + somatório de 1 até comp camada entrada (entrada_i * w_ij)
        z_j = f (z_in_j)

        camada de saída:
        y_in_k = b_0k + somatório de 1 até comp camada oculta (z_j * bw_jk)
        y_k = f(y_in_k)
        """
        entrada = np.array(entrada)
        # Camada oculta:

        # z_in_j = w_0j + somatório de cada entrada multiplicado pelo
        # respectivo peso somado ao bias do neurônio j da camada oculta
        self.z_in = self.w0 + self.W.T @ entrada

        # aplicamos a função de ativação em z_in_j
        funcao_de_ativacao_vec = np.vectorize(self.funcao_de_ativacao)
        self.z = funcao_de_ativacao_vec(self.z_in)

        # Camada de saída

        # y_in_k = b_0k + somatório de 1 até comp camada oculta (z_j * bw_jk)
        self.y_in = self.b0 + self.B.T @ self.z

        # y_k = f(y_in_k)
        self.y = funcao_de_ativacao_vec(self.y_in)

        self.logger.log_entrada(list(entrada))
        self.logger.log_camada_oculta(self.W, self.w0, self.z_in, self.z)
        self.logger.log_camada_saida(self.B, self.b0, self.y_in, self.y)

    def teste(self, dados, rotulos, letras, valor_esperado):
        self.logger.log_inicio_teste(dados.shape[0])
        count = 0
        resultados = []
        for i in range(dados.shape[0]):
            self.logger.log_iteracao_teste(i)
            self.forward(dados.iloc[i])
            saida = np.array(self.y)
            saida = list(np.round(saida, 2))
            indice_letra = saida.index(max(saida))

            prev = letras[indice_letra]
            esp = valor_esperado.iat[i, 0]

            # Casefold normaliza String
            if (prev.casefold() == esp.casefold()):
                count = count + 1

            resp = rotulos[i]
            erro = np.array(self.y) - np.array(resp)
            erro_total = sum(erro)

            self.logger.log_resultado_teste(prev, esp, saida, erro_total)

            resultados.append({
                'esperado': esp,
                'previsto': prev,
                'saida': saida,
                'erro': erro_total
            })

        self.logger.log_resumo_teste(resultados, letras)
        self.logger.log_acuracia_final(count, dados.shape[0])

        self.writer.write_saidas_teste(resultados)
        self.writer.write_acuracia(count, dados.shape[0])
        return resultados

    def backpropagate(self, x, t, z_in, z, y_in, y):
        """
        CAMADA DE SAÍDA - Cálculo do termo de erro:
            δ_k = (t_k - y_k) * f'(y_in_k)

        CAMADA DE SAÍDA - Cálculo das correções de peso:
            Δb_jk = α * δ_k * z_j
            Δb_0k = α * δ_k

        CAMADA OCULTA - Retropropagação do erro:
            δ_in_j = B * δ_k
            δ_j    = δ_in_j * f'(z_in_j)

        CAMADA OCULTA - Cálculo das correções de peso:
            ΔW   = α * δ_j * x
            Δw_0j = α * δ_j

        ATUALIZAÇÃO DOS PESOS:
            b_jk(new) = b_jk(old) + Δb_jk
            w_ij(new) = w_ij(old) + Δw_ij
        """

        # Converte entradas para arrays NumPy para operações matriciais
        x = np.array(x)
        t = np.array(t)
        z = np.array(z)
        z_in = np.array(z_in)
        y = np.array(y)
        y_in = np.array(y_in)

        taxa = self.taxa_de_aprendizado

        # Vetoriza a derivada da sigmoid para aplicação elemento a elemento
        derivada_sigmoid_vec = np.vectorize(derivada_sigmoid)

        # δ_k = (t_k - y_k) * f'(y_in_k)
        deltaMaior_k = (t - y) * derivada_sigmoid_vec(y_in)

        # Δb_jk = α * δ_k * z_j
        delta_b_jk = taxa * np.outer(z, deltaMaior_k)

        # Δb_0k = α * δ_k
        # Correção dos bias da camada de saída: shape (n_saida,)
        delta_b0 = taxa * deltaMaior_k

        # δ_in_j = B * δ_k
        deltaMaior_in_j = self.B @ deltaMaior_k

        # δ_j = δ_in_j * f'(z_in_j)
        deltaMaior_j = deltaMaior_in_j * derivada_sigmoid_vec(z_in)

        # ΔW = α * δ_j * x
        delta_W = taxa * np.outer(x, deltaMaior_j)

        # Δw_0j = α * δ_j
        delta_w0 = taxa * deltaMaior_j

        # Atualização dos pesos

        # b_jk(new) = b_jk(old) + Δb_jk
        self.B += delta_b_jk

        # b_0k(new) = b_0k(old) + Δb_0k
        self.b0 += delta_b0

        # w_ij(new) = w_ij(old) + ΔW
        self.W += delta_W

        # w_0j(new) = w_0j(old) + Δw_0j
        self.w0 += delta_w0

        self.logger.log_backprop_erros(y, t, sum((t[k] - y[k]) for k in range(self.comprimento_saida)))
        self.logger.log_erro_saida(deltaMaior_k, delta_b_jk, delta_b0)
        self.logger.log_erro_oculta(deltaMaior_in_j, deltaMaior_j, delta_W, delta_w0)
        self.logger.log_atualizacao_pesos(self.B, self.b0, self.W, self.w0)

        # Erro instantâneo E(n) DESTA amostra, na convenção de Haykin:
        # E(n) = (1/2) * Σ_k (t_k - y_k)^2, somando sobre os m neurônios de saída.
        # OBS: o fator não afeta o aprendizado, pois o gradiente usa (t-y) direto
        # (sem o 1/2); ele apenas alinha o valor reportado à definição dos slides.
        # O E_av = (1/N) Σ_n E(n) é obtido em fit() ao fazer a média desses valores
        # por época, fechando exatamente a fórmula E_av do material.
        m = self.comprimento_saida
        erro = 0.5 * np.sum((t[:m] - y[:m]) ** 2)
        return erro

    def set_funcao_de_ativacao(self, funcao):
        self.funcao_de_ativacao = funcao

    def matriz_confusao(self, resultados, letras):
        """
        Essa função visa gerar e exibir no console a matriz de confusão após o teste
        """

        n = len(letras)
        matriz = [[0] * n for _ in range(n)]

        for res in resultados:
            indice_real = letras.index(res['esperado'])
            indice_previsto = letras.index(res['previsto'])
            matriz[indice_real][indice_previsto] += 1

        # Impressão no console
        largura_da_coluna = 4
        print("\n" + "=" * 60)
        print("MATRIZ DE CONFUSÃO".center(60))
        print("=" * 60)

        cabecalho = "    " + "".join(l.center(largura_da_coluna) for l in letras)
        print(cabecalho)
        print("-" * (largura_da_coluna * n + 5))

        for i in range(n):
            linha = letras[i] + " | " + "".join(str(matriz[i][j]).center(largura_da_coluna) for j in range(n))
            print(linha)

        print("-" * 60)
        self.writer.write_matriz_confusao(matriz, letras)

        # Heatmap da matriz de confusão para os slides
        caminho_mc = self.writer._obter_caminho("9_matriz_confusao.png")
        plots.plotar_matriz_confusao(matriz, letras, caminho_saida=caminho_mc)

        return matriz