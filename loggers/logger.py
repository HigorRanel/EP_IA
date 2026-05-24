"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

Módulo de que implementa funções para permitir
a visualização da execução e resultados do modelo
no terminal
"""

import sys
import time
import numpy as np
from collections import Counter
import pandas as pd


class Colors:
    """Códigos ANSI para colorir e formatar o terminal"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    GREEN = '\033[92m'
    RED = '\033[91m'


class Logger:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.traco_fino = "-" * 50
        self.traco_grosso = "=" * 60
        # Estado da barra de progresso
        self._epoca_inicio = None

    def formatar(self, dado):
        """Identifica vetores e matrizes, trunca as grandes e aplica indentação"""
        if hasattr(dado, 'tolist'):
            dado = dado.tolist()

        if isinstance(dado, list):
            if len(dado) > 0 and isinstance(dado[0], list):
                num_linhas = len(dado)
                num_cols = len(dado[0])

                linhas_formatadas = [f"[Matriz {num_linhas}x{num_cols}]"]
                max_linhas_exibicao = 3

                for i, linha in enumerate(dado):
                    if i < max_linhas_exibicao:
                        linha_str = str(np.round(np.array(linha, dtype=float), 4).tolist())
                        linhas_formatadas.append("    " + linha_str)
                    elif i == max_linhas_exibicao:
                        linhas_formatadas.append("... (matriz truncada)")
                        break

                return "\n".join(linhas_formatadas)

            return "    " + str(np.round(np.array(dado, dtype=float), 4).tolist())

        return "    " + str(round(dado, 4) if isinstance(dado, float) else dado)

    def log_configuracoes_iniciais(self, n_entradas, n_ocultas, n_saidas, taxa_aprendizado, ativacao):
        print(f"{Colors.BOLD}{self.traco_grosso}")
        print(" INICIALIZAÇÃO DA MLP ".center(60, " "))
        print(self.traco_grosso)
        print(f"Arquitetura: {n_entradas} (Entrada) | {n_ocultas} (Oculta) | {n_saidas} (Saída)")
        print(f"Taxa de Aprendizado: {taxa_aprendizado}")
        print(f"Função de Ativação: {ativacao}")
        print(f"{self.traco_grosso}{Colors.RESET}\n")

    def barra_progresso_epoca(self, epoca, total_epocas, atual, total,
                              erro_medio=None, largura=30):
        """
        Imprime/atualiza uma barra de progresso de uma época
        No modo verbose, não faz nada
        """
        if self.verbose:
            return

        if atual <= 1 or self._epoca_inicio is None:
            self._epoca_inicio = time.time()
            print(f"{Colors.BOLD}Época {epoca + 1}/{total_epocas}{Colors.RESET}")

        fracao = atual / total
        preenchido = int(largura * fracao)
        if preenchido >= largura:
            barra = "━" * largura
        else:
            barra = "━" * preenchido + "╸" + " " * (largura - preenchido - 1)

        decorrido = time.time() - self._epoca_inicio
        passo_ms = (decorrido / atual * 1000) if atual else 0.0

        msg = (f"\r{atual:>5}/{total} "
               f"{Colors.CYAN}{barra}{Colors.RESET} "
               f"{decorrido:4.0f}s {passo_ms:3.0f}ms/passo")
        if erro_medio is not None:
            msg += f" - erro: {erro_medio:.6f}"

        sys.stdout.write(msg)
        sys.stdout.flush()

        if atual >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._epoca_inicio = None

    def log_erro_validacao(self, epoca, erro_treino, erro_val):
        """
        Ao fim de uma época, mostra o EQM de treino e de validação lado a la
        Sempre visível
        """
        print(f"{Colors.CYAN}   Época {epoca}: "
              f"EQM_treino={erro_treino:.6f}  |  EQM_val={erro_val:.6f}{Colors.RESET}")

    def log_parada_antecipada(self, epoca, erro_medio, limiar):
        print(f"{Colors.BOLD}{Colors.GREEN}"
              f"Parada antecipada na época {epoca}: "
              f"erro {round(erro_medio, 6)} <= limiar {limiar}{Colors.RESET}")

    def log_parada_antecipada_val(self, epoca, erro_val, paciencia):
        """Avisa que a parada antecipada por validação foi acionada."""
        print(f"\n{Colors.BOLD}{Colors.GREEN}"
              f"Parada antecipada na época {epoca}: erro de validação não "
              f"melhora há {paciencia} épocas (EQM_val={round(erro_val, 6)}).{Colors.RESET}")

    def log_inicio_teste(self, n_teste):
        print(f"\n{Colors.BOLD}{Colors.BLUE}{self.traco_grosso}")
        print(" TESTE ".center(60))
        print(self.traco_grosso)
        print(f"Amostras de teste: {n_teste}")
        print(f"{self.traco_grosso}{Colors.RESET}")

    def log_acuracia_final(self, count, total):
        pct = round(count / total * 100, 2)
        print(f"\n{Colors.BOLD}{Colors.GREEN}{self.traco_grosso}")
        print("RESULTADO FINAL ".center(60))
        print(f"Acurácia: {count}/{total} = {pct}%")
        print(f"{self.traco_grosso}{Colors.RESET}\n")

    def log_resumo_teste(self, resultados, letras):
        """
        Resumo dos resultados de teste

        Mostra:
          - acurácia por classe (acertos/total de cada letra)
          - a lista compacta dos erros cometidos (esperado - previsto)
        """
        # Contagens por classe
        total_por_classe = {l: 0 for l in letras}
        acertos_por_classe = {l: 0 for l in letras}
        erros = []  # (esperado, previsto)

        for res in resultados:
            esp = res['esperado']
            prev = res['previsto']
            if esp in total_por_classe:
                total_por_classe[esp] += 1
                if str(esp).casefold() == str(prev).casefold():
                    acertos_por_classe[esp] += 1
                else:
                    erros.append((esp, prev))

        # Acurácia por classe
        print(f"\n{Colors.BOLD}{Colors.BLUE}{self.traco_fino}")
        print(" ACURÁCIA POR CLASSE ".center(50))
        print(f"{self.traco_fino}{Colors.RESET}")
        for l in letras:
            tot = total_por_classe[l]
            if tot == 0:
                continue  # classe ausente no conjunto de teste
            ac = acertos_por_classe[l]
            pct = ac / tot * 100
            # Cor: verde se 100%, vermelho se errou alguma
            cor = Colors.GREEN if ac == tot else Colors.RED
            barra = "█" * int(pct / 10)  # mini-barra visual (0 a 10 blocos)
            print(f"{l}: {cor}{ac:>3}/{tot:<3} ({pct:5.1f}%){Colors.RESET} {barra}")

        # Lista de erros
        print(f"\n{Colors.BOLD}{Colors.RED}{self.traco_fino}")
        print(f" ERROS COMETIDOS: {len(erros)} ".center(50))
        print(f"{self.traco_fino}{Colors.RESET}")

        # Agrupa erros idênticos (esperado, previsto) e conta ocorrências
        contagem = Counter(erros)
        # Ordena do par mais frequente para o menos frequente
        pares_ordenados = contagem.most_common()

        max_linhas = 30  # evita encher a tela em conjuntos grandes
        for (esp, prev), n in pares_ordenados[:max_linhas]:
            vezes = f" ({n}x)" if n > 1 else ""
            print(f"Esperado {Colors.BOLD}{esp}{Colors.RESET} "
                  f"previsto {Colors.RED}{prev}{Colors.RESET}{vezes}")
        restantes = len(pares_ordenados) - max_linhas
        if restantes > 0:
            print(f"e mais {restantes} tipo(s) de erro "
                  f"(ver arquivo 5_saidas_teste.csv para a lista completa).")

    def log_inicio_epoca(self, epoca, total_epocas):
        if not self.verbose:
            return
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{self.traco_grosso}")
        print(f" ÉPOCA {epoca + 1} / {total_epocas} ".center(60))
        print(f"{self.traco_grosso}{Colors.RESET}")

    def log_iteracao_dado(self, indice_dado):
        if not self.verbose:
            return
        print(f"\n{Colors.BOLD}{Colors.BLUE}{self.traco_fino}")
        print(f" DADO DE TREINO #{indice_dado}")
        print(f"{self.traco_fino}{Colors.RESET}")

    def log_entrada(self, entrada):
        if not self.verbose:
            return
        print(f"\n{Colors.BOLD}{Colors.BLUE}-[FEEDFORWARD]-{Colors.RESET}")
        print(f"{Colors.BOLD}  Entrada (x):{Colors.RESET}")
        print(self.formatar(entrada))

    def log_camada_oculta(self, W, w0, z_in, z):
        if not self.verbose:
            return
        print(f"\n{Colors.BOLD}{Colors.CYAN}[Camada Oculta]{Colors.RESET}")
        print(f"{Colors.BOLD}Pesos Entrada Oculta (W):{Colors.RESET}")
        print(self.formatar(W))
        print(f"{Colors.BOLD}Bias Ocultos (w0):{Colors.RESET}")
        print(self.formatar(w0))
        print(f"{Colors.BOLD} Entrada dos neurônios (z_in):{Colors.RESET}")
        print(self.formatar(z_in))
        print(f"{Colors.BOLD}Saída dos neurônios (z):{Colors.RESET}")
        print(self.formatar(z))

    def log_camada_saida(self, B, b0, y_in, y):
        if not self.verbose:
            return
        print(f"\n{Colors.BOLD}{Colors.CYAN}[Camada de Saída]{Colors.RESET}")
        print(f"{Colors.BOLD}Pesos Oculta Saída (B):{Colors.RESET}")
        print(self.formatar(B))
        print(f"{Colors.BOLD}Bias de Saída (b0):{Colors.RESET}")
        print(self.formatar(b0))
        print(f"{Colors.BOLD}Entrada dos neurônios (y_in):{Colors.RESET}")
        print(self.formatar(y_in))
        print(f"{Colors.BOLD}Saída final (y):{Colors.RESET}")
        print(self.formatar(y))

    def log_backprop_erros(self, y, t, erro):
        if not self.verbose:
            return
        print(f"\n{Colors.BOLD}{Colors.BLUE}-[BACKPROPAGATION]-{Colors.RESET}")
        print(f"{Colors.BOLD}Saída Prevista (y):{Colors.RESET}")
        print(self.formatar(y))
        print(f"{Colors.BOLD}Saída Esperada (t):{Colors.RESET}")
        print(self.formatar(t))
        print(f"{Colors.BOLD}Erro Calculado:{Colors.RESET}")
        print(self.formatar(erro))

    def log_erro_saida(self, delta_k, delta_w_jk, delta_w_0k):
        if not self.verbose:
            return
        print(f"\n{Colors.BOLD}{Colors.CYAN}[Erro e Correção Camada de Saída]{Colors.RESET}")
        print(f"{Colors.BOLD}Termo de erro (δ_k):{Colors.RESET}")
        print(self.formatar(delta_k))
        print(f"{Colors.BOLD}Correção de pesos (Δw_jk):{Colors.RESET}")
        print(self.formatar(delta_w_jk))
        print(f"{Colors.BOLD}Correção de bias (Δw_0k):{Colors.RESET}")
        print(self.formatar(delta_w_0k))

    def log_erro_oculta(self, delta_in_j, delta_j, delta_v_ij, delta_v_0j):
        if not self.verbose:
            return
        print(f"\n{Colors.BOLD}{Colors.CYAN}[Erro e Correção  Camada Oculta]{Colors.RESET}")
        print(f"{Colors.BOLD}Soma das entradas de erro (δ_in_j):{Colors.RESET}")
        print(self.formatar(delta_in_j))
        print(f"{Colors.BOLD}Termo de erro (δ_j):{Colors.RESET}")
        print(self.formatar(delta_j))
        print(f"{Colors.BOLD}Correção de pesos (Δv_ij):{Colors.RESET}")
        print(self.formatar(delta_v_ij))
        print(f"{Colors.BOLD}Correção de bias (Δv_0j):{Colors.RESET}")
        print(self.formatar(delta_v_0j))

    def log_atualizacao_pesos(self, w_jk_new, w_0k_new, v_ij_new, v_0j_new):
        if not self.verbose:
            return
        print(f"\n{Colors.BOLD}{Colors.CYAN}[Pesos e Bias Atualizados]{Colors.RESET}")
        print(f"{Colors.BOLD}Novos pesos Oculta Saída (B_new):{Colors.RESET}")
        print(self.formatar(w_jk_new))
        print(f"{Colors.BOLD}Novos bias de Saída (b0_new):{Colors.RESET}")
        print(self.formatar(w_0k_new))
        print(f"{Colors.BOLD}Novos pesos Entrada Oculta (W_new):{Colors.RESET}")
        print(self.formatar(v_ij_new))
        print(f"{Colors.BOLD}Novos bias Ocultos (w0_new):{Colors.RESET}")
        print(self.formatar(v_0j_new))
        print("\n")

    def log_iteracao_teste(self, indice_dado):
        if not self.verbose:
            return
        print(f"\n{Colors.BOLD}{Colors.BLUE}{self.traco_grosso}")
        print(f"TESTANDO DADO #{indice_dado} ".center(60, "-"))
        print(f"{self.traco_grosso}{Colors.RESET}")

    def log_resultado_teste(self, previsto, esperado, saida_rede, erro):
        if not self.verbose:
            return
        print(f"\n{Colors.BOLD}[RESULTADO DO TESTE]{Colors.RESET}")
        cor_resultado = Colors.GREEN if str(previsto).casefold() == str(esperado).casefold() else Colors.RED

        print(f"{Colors.BOLD}Previsto: {cor_resultado}{previsto}{Colors.RESET}")
        print(f"{Colors.BOLD}Resposta: {esperado}{Colors.RESET}")
        print(f"{Colors.BOLD}Saída da Rede:{Colors.RESET}")
        print(self.formatar(saida_rede))
        print(f"{Colors.BOLD}Erro Total da Amostra:{Colors.RESET}")
        print(self.formatar(erro))