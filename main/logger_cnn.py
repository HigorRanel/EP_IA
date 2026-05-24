"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger import Logger, Colors


class LoggerCNN(Logger):

    # ------------------------------------------------------------------
    # INICIALIZAÇÃO E CONFIGURAÇÃO
    # ------------------------------------------------------------------

    def log_configuracoes_cnn(self, tarefa, modo_dados, config):
        """
        Exibe no console os hiperparâmetros da CNN antes do treinamento.
        """
        print(f"{Colors.BOLD}{self.traco_grosso}")
        print(f" INICIALIZAÇÃO DA CNN ".center(60))
        print(self.traco_grosso)
        print(f"Tarefa: {tarefa.upper()}")
        print(f"Modo de dados: {modo_dados.upper()}")
        for chave, valor in config.items():
            print(f"{chave:<22} {valor}")
        print(f"{self.traco_grosso}{Colors.RESET}\n")

    def log_inicio_experimento(self, tarefa, modo_dados):
        """
        Cabeçalho visual no início de cada um dos 4 experimentos.
        """
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'#' * 60}")
        print(f"EXPERIMENTO: {tarefa.upper()} | {modo_dados.upper()}".center(60))
        print(f"{'#' * 60}{Colors.RESET}")

    def log_extracao_descritores(self, tipo, n_amostras, dim_saida = None):
        """
        Informa o início e o resultado da extração de descritores HOG/LBP.
        """
        if dim_saida is None:
            print(f"\n{Colors.BOLD}{Colors.CYAN}"
                  f"---[EXTRAÇÃO DE DESCRITORES: {tipo}]---{Colors.RESET}")
            print(f"Processando {n_amostras} imagens...")
        else:
            print(f"{Colors.BOLD}{Colors.GREEN}"
                  f"{tipo} extraído — {n_amostras} amostras | dim={dim_saida}"
                  f"{Colors.RESET}")

    # ------------------------------------------------------------------
    # TREINAMENTO
    # ------------------------------------------------------------------

    def log_inicio_treinamento(self, tarefa, modo_dados,
                                n_treino, n_val, epocas):
        """
        Cabeçalho do bloco de treinamento.
        """
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{self.traco_grosso}")
        print(f"TREINAMENTO — {tarefa.upper()} | {modo_dados.upper()}".center(60, "="))
        print(self.traco_grosso)
        print(f"mostras treino: {n_treino}")
        print(f"Amostras validação: {n_val}")
        print(f"Épocas máx.: {epocas}")
        print(f"{self.traco_grosso}{Colors.RESET}")

    # ------------------------------------------------------------------
    # TESTE
    # ------------------------------------------------------------------

    def log_inicio_teste_cnn(self, tarefa, modo_dados, n_teste):
        """
        Cabeçalho do bloco de teste.
        """
        print(f"\n{Colors.BOLD}{Colors.BLUE}{self.traco_grosso}")
        print(f"TESTE — {tarefa.upper()} | {modo_dados.upper()} ".center(60, "="))
        print(self.traco_grosso)
        print(f"Amostras de teste: {n_teste}")
        print(f"{self.traco_grosso}{Colors.RESET}")

    def log_acuracia_final(self, acertos, total, tarefa, modo_dados):
        """
        Exibe a acurácia final do experimento com destaque visual.

        acertos: número de predições corretas.
        total: total de amostras testadas.
        tarefa:'multiclasse' ou 'binaria'.
        modo_dados: 'bruto' ou 'hog_lbp'.
        """
        pct = round(acertos / total * 100, 2)
        print(f"\n{Colors.BOLD}{Colors.GREEN}{self.traco_grosso}")
        print(f"RESULTADO FINAL — {tarefa.upper()} | {modo_dados.upper()} ".center(60))
        print(f"Acurácia: {acertos}/{total} = {pct}%")
        print(f"{self.traco_grosso}{Colors.RESET}\n")