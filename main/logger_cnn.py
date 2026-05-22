"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

LoggerCNN: extensão do Logger base para logs de console da CNN.

Herda de Logger toda a infraestrutura de cores (ANSI), separadores
(traco_fino, traco_grosso) e o método _formatar, adicionando métodos
com vocabulário próprio da CNN:
  - log_configuracoes_cnn
  - log_inicio_experimento
  - log_extracao_descritores
  - log_inicio_epoca_cnn
  - log_resultado_epoca
  - log_inicio_teste_cnn
  - log_resultado_teste_cnn
  - log_acuracia_final
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import Logger, Colors


class LoggerCNN(Logger):
    """
    Subclasse de Logger com métodos de log voltados para a CNN.

    Herda:
        - Colors: códigos ANSI para colorir o terminal
        - traco_fino / traco_grosso: separadores visuais
        - _formatar: formatação de arrays e escalares
    """

    # ------------------------------------------------------------------
    # INICIALIZAÇÃO E CONFIGURAÇÃO
    # ------------------------------------------------------------------

    def log_configuracoes_cnn(self, tarefa: str, modo_dados: str, config: dict):
        """
        Exibe no console os hiperparâmetros da CNN antes do treinamento.

        Args:
            tarefa:     'multiclasse' ou 'binaria'.
            modo_dados: 'bruto' ou 'hog_lbp'.
            config:     dicionário com os hiperparâmetros.
        """
        print(f"{Colors.BOLD}{self.traco_grosso}")
        print(f" INICIALIZAÇÃO DA CNN ".center(60))
        print(self.traco_grosso)
        print(f" ➔ Tarefa:           {tarefa.upper()}")
        print(f" ➔ Modo de dados:    {modo_dados.upper()}")
        for chave, valor in config.items():
            print(f" ➔ {chave:<22} {valor}")
        print(f"{self.traco_grosso}{Colors.RESET}\n")

    def log_inicio_experimento(self, tarefa: str, modo_dados: str):
        """
        Cabeçalho visual no início de cada um dos 4 experimentos.

        Args:
            tarefa:     'multiclasse' ou 'binaria'.
            modo_dados: 'bruto' ou 'hog_lbp'.
        """
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'#' * 60}")
        print(f"  EXPERIMENTO: {tarefa.upper()} | {modo_dados.upper()}".center(60))
        print(f"{'#' * 60}{Colors.RESET}")

    def log_extracao_descritores(self, tipo: str, n_amostras: int, dim_saida: int = None):
        """
        Informa o início e o resultado da extração de descritores HOG/LBP.

        Args:
            tipo:       'HOG', 'LBP' ou 'HOG+LBP'.
            n_amostras: número de imagens processadas.
            dim_saida:  dimensão do vetor resultante (None durante o início).
        """
        if dim_saida is None:
            print(f"\n{Colors.BOLD}{Colors.CYAN}"
                  f"--- [EXTRAÇÃO DE DESCRITORES: {tipo}] ---{Colors.RESET}")
            print(f" ➔ Processando {n_amostras} imagens...")
        else:
            print(f"{Colors.BOLD}{Colors.GREEN}"
                  f" ✓ {tipo} extraído — {n_amostras} amostras | dim={dim_saida}"
                  f"{Colors.RESET}")

    # ------------------------------------------------------------------
    # TREINAMENTO
    # ------------------------------------------------------------------

    def log_inicio_treinamento(self, tarefa: str, modo_dados: str,
                                n_treino: int, n_val: int, epocas: int):
        """
        Cabeçalho do bloco de treinamento.

        Args:
            tarefa:     'multiclasse' ou 'binaria'.
            modo_dados: 'bruto' ou 'hog_lbp'.
            n_treino:   número de amostras de treino.
            n_val:      número de amostras de validação.
            epocas:     número máximo de épocas configurado.
        """
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{self.traco_grosso}")
        print(f" TREINAMENTO — {tarefa.upper()} | {modo_dados.upper()} ".center(60, "="))
        print(self.traco_grosso)
        print(f" ➔ Amostras treino:  {n_treino}")
        print(f" ➔ Amostras val:     {n_val}")
        print(f" ➔ Épocas máx.:      {epocas}")
        print(f"{self.traco_grosso}{Colors.RESET}")

    def log_inicio_epoca_cnn(self, epoca: int, total_epocas: int):
        """
        Separador de época — equivalente ao log_inicio_epoca do MLP,
        mas sem assumir vocabulário de gradiente descendente manual.

        Args:
            epoca:        índice da época atual (0-based).
            total_epocas: total de épocas configurado.
        """
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{self.traco_fino}")
        print(f" ÉPOCA {epoca + 1} / {total_epocas} ".center(50, "-"))
        print(f"{self.traco_fino}{Colors.RESET}")

    def log_resultado_epoca(self, epoca: int, loss: float, acc: float,
                             val_loss: float = None, val_acc: float = None):
        """
        Exibe as métricas ao final de cada época.

        Args:
            epoca:    índice da época (0-based).
            loss:     loss de treinamento.
            acc:      acurácia de treinamento.
            val_loss: loss de validação (None se não houver).
            val_acc:  acurácia de validação (None se não houver).
        """
        print(f"{Colors.BOLD}{Colors.CYAN} ➔ Época {epoca + 1}: "
              f"loss={round(loss, 4)}  acc={round(acc * 100, 2)}%", end="")
        if val_loss is not None:
            print(f"  |  val_loss={round(val_loss, 4)}  val_acc={round(val_acc * 100, 2)}%",
                  end="")
        print(Colors.RESET)

    def log_parada_antecipada(self, epoca: int):
        """
        Avisa quando o EarlyStopping interrompeu o treinamento.

        Args:
            epoca: época em que ocorreu a parada.
        """
        print(f"\n{Colors.BOLD}{Colors.GREEN}"
              f" ✓ EarlyStopping: treinamento interrompido na época {epoca}."
              f"{Colors.RESET}")

    # ------------------------------------------------------------------
    # TESTE
    # ------------------------------------------------------------------

    def log_inicio_teste_cnn(self, tarefa: str, modo_dados: str, n_teste: int):
        """
        Cabeçalho do bloco de teste.

        Args:
            tarefa:     'multiclasse' ou 'binaria'.
            modo_dados: 'bruto' ou 'hog_lbp'.
            n_teste:    número de amostras de teste.
        """
        print(f"\n{Colors.BOLD}{Colors.BLUE}{self.traco_grosso}")
        print(f" TESTE — {tarefa.upper()} | {modo_dados.upper()} ".center(60, "="))
        print(self.traco_grosso)
        print(f" ➔ Amostras de teste: {n_teste}")
        print(f"{self.traco_grosso}{Colors.RESET}")

    def log_resultado_teste_cnn(self, indice: int, esperado: str,
                                 previsto: str, proba: list):
        """
        Exibe o resultado de uma amostra de teste individual.

        Equivalente ao log_resultado_teste do Logger, mas com
        nomenclatura CNN (probabilidade em vez de erro quadrático).

        Args:
            indice:   índice da amostra no conjunto de teste.
            esperado: nome da classe verdadeira.
            previsto: nome da classe prevista.
            proba:    vetor de probabilidades (softmax ou sigmoid).
        """
        cor = Colors.GREEN if esperado == previsto else Colors.RED
        print(f"\n{Colors.BOLD}{Colors.BLUE}{self.traco_fino}")
        print(f" AMOSTRA #{indice}{Colors.RESET}")
        print(f"{Colors.BOLD} ➔ Esperado: {esperado}")
        print(f" ➔ Previsto: {cor}{previsto}{Colors.RESET}")
        print(f"{Colors.BOLD} ➔ Probabilidades:{Colors.RESET}")
        print(self._formatar(proba))

    def log_acuracia_final(self, acertos: int, total: int, tarefa: str, modo_dados: str):
        """
        Exibe a acurácia final do experimento com destaque visual.

        Args:
            acertos:    número de predições corretas.
            total:      total de amostras testadas.
            tarefa:     'multiclasse' ou 'binaria'.
            modo_dados: 'bruto' ou 'hog_lbp'.
        """
        pct = round(acertos / total * 100, 2)
        print(f"\n{Colors.BOLD}{Colors.GREEN}{self.traco_grosso}")
        print(f" RESULTADO FINAL — {tarefa.upper()} | {modo_dados.upper()} ".center(60))
        print(f" ➔ Acurácia: {acertos}/{total} = {pct}%")
        print(f"{self.traco_grosso}{Colors.RESET}\n")
