import numpy as np
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
    def __init__(self):
        self.traco_fino = "-" * 50
        self.traco_grosso = "=" * 60

    def _formatar(self, dado):
        """Identifica vetores e matrizes, trunca as grandes e aplica recuo (indentação)."""
        if hasattr(dado, 'tolist'):
            dado = dado.tolist()

        if isinstance(dado, list):
            # Se for uma MATRIZ (lista de listas 2D)
            if len(dado) > 0 and isinstance(dado[0], list):
                num_linhas = len(dado)
                num_cols = len(dado[0])

                # Sem cor para o indicador de matriz
                linhas_formatadas = [f"    [Matriz {num_linhas}x{num_cols}]"]
                max_linhas_exibicao = 3

                for i, linha in enumerate(dado):
                    if i < max_linhas_exibicao:
                        linha_str = str(np.round(np.array(linha, dtype=float), 4).tolist())
                        linhas_formatadas.append("    " + linha_str)
                    elif i == max_linhas_exibicao:
                        linhas_formatadas.append("    ... (matriz truncada)")
                        break

                return "\n".join(linhas_formatadas)

            # Se for um VETOR (1D)
            return "    " + str(np.round(np.array(dado, dtype=float), 4).tolist())

        return "    " + str(round(dado, 4) if isinstance(dado, float) else dado)

    # ==========================================
    # INICIALIZAÇÃO E CONTROLE DE FLUXO
    # ==========================================
    def log_configuracoes_iniciais(self, n_entradas, n_ocultas, n_saidas, taxa_aprendizado, ativacao):
        print(f"{Colors.BOLD}{self.traco_grosso}")
        print(" INICIALIZAÇÃO DA MLP ".center(60, " "))
        print(self.traco_grosso)
        print(f" ➔ Arquitetura:         {n_entradas} (Entrada) -> {n_ocultas} (Oculta) -> {n_saidas} (Saída)")
        print(f" ➔ Taxa de Aprendizado: {taxa_aprendizado}")
        print(f" ➔ Função de Ativação:  {ativacao}")
        print(f"{self.traco_grosso}{Colors.RESET}\n")

    def log_inicio_epoca(self, epoca, total_epocas):
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{self.traco_grosso}")
        print(f" ÉPOCA {epoca + 1} / {total_epocas} ".center(60, "="))
        print(f"{self.traco_grosso}{Colors.RESET}")

    def log_iteracao_dado(self, indice_dado):
        print(f"\n{Colors.BOLD}{Colors.BLUE}{self.traco_fino}")
        print(f" DADO DE TREINO #{indice_dado}")
        print(f"{self.traco_fino}{Colors.RESET}")

    # ==========================================
    # FEEDFORWARD
    # ==========================================
    def log_entrada(self, entrada):
        print(f"\n{Colors.BOLD}{Colors.BLUE}--- [FEEDFORWARD] ---{Colors.RESET}")
        print(f"{Colors.BOLD} ➔ Entrada (x):{Colors.RESET}")
        print(self._formatar(entrada))

    def log_camada_oculta(self, W, z_in, z):
        print(f"\n{Colors.BOLD}{Colors.CYAN}[Camada Oculta]{Colors.RESET}")
        print(f"{Colors.BOLD} ➔ Pesos Entrada->Oculta (W):{Colors.RESET}")
        print(self._formatar(W))
        print(f"{Colors.BOLD} ➔ Entrada dos neurônios (z_in):{Colors.RESET}")
        print(self._formatar(z_in))
        print(f"{Colors.BOLD} ➔ Saída dos neurônios (z):{Colors.RESET}")
        print(self._formatar(z))

    def log_camada_saida(self, B, y_in, y):
        print(f"\n{Colors.BOLD}{Colors.CYAN}[Camada de Saída]{Colors.RESET}")
        print(f"{Colors.BOLD} ➔ Pesos Oculta->Saída (B):{Colors.RESET}")
        print(self._formatar(B))
        print(f"{Colors.BOLD} ➔ Entrada dos neurônios (y_in):{Colors.RESET}")
        print(self._formatar(y_in))
        print(f"{Colors.BOLD} ➔ Saída final (y):{Colors.RESET}")
        print(self._formatar(y))

    # ==========================================
    # BACKPROPAGATION
    # ==========================================
    def log_backprop_erros(self, y, t, erro):
        print(f"\n{Colors.BOLD}{Colors.BLUE}--- [BACKPROPAGATION] ---{Colors.RESET}")
        print(f"{Colors.BOLD} ➔ Saída Prevista (y):{Colors.RESET}")
        print(self._formatar(y))
        print(f"{Colors.BOLD} ➔ Saída Esperada (t):{Colors.RESET}")
        print(self._formatar(t))
        print(f"{Colors.BOLD} ➔ Erro Calculado:{Colors.RESET}")
        print(self._formatar(erro))

    def log_erro_saida(self, delta_k, delta_w_jk, delta_w_0k):
        print(f"\n{Colors.BOLD}{Colors.CYAN}[Erro e Correção - Camada de Saída]{Colors.RESET}")
        print(f"{Colors.BOLD} ➔ Termo de erro (δ_k):{Colors.RESET}")
        print(self._formatar(delta_k))
        print(f"{Colors.BOLD} ➔ Correção de pesos (Δw_jk):{Colors.RESET}")
        print(self._formatar(delta_w_jk))
        print(f"{Colors.BOLD} ➔ Correção de bias (Δw_0k):{Colors.RESET}")
        print(self._formatar(delta_w_0k))

    def log_erro_oculta(self, delta_in_j, delta_j, delta_v_ij, delta_v_0j):
        print(f"\n{Colors.BOLD}{Colors.CYAN}[Erro e Correção - Camada Oculta]{Colors.RESET}")
        print(f"{Colors.BOLD} ➔ Soma das entradas de erro (δ_in_j):{Colors.RESET}")
        print(self._formatar(delta_in_j))
        print(f"{Colors.BOLD} ➔ Termo de erro (δ_j):{Colors.RESET}")
        print(self._formatar(delta_j))
        print(f"{Colors.BOLD} ➔ Correção de pesos (Δv_ij):{Colors.RESET}")
        print(self._formatar(delta_v_ij))
        print(f"{Colors.BOLD} ➔ Correção de bias (Δv_0j):{Colors.RESET}")
        print(self._formatar(delta_v_0j))

    def log_atualizacao_pesos(self, w_jk_new, w_0k_new, v_ij_new, v_0j_new):
        print(f"\n{Colors.BOLD}{Colors.CYAN}[Pesos e Bias Atualizados]{Colors.RESET}")
        print(f"{Colors.BOLD} ➔ Novos pesos Oculta->Saída (B_new):{Colors.RESET}")
        print(self._formatar(w_jk_new))
        print(f"{Colors.BOLD} ➔ Novos bias de Saída (b0_new):{Colors.RESET}")
        print(self._formatar(w_0k_new))
        print(f"{Colors.BOLD} ➔ Novos pesos Entrada->Oculta (W_new):{Colors.RESET}")
        print(self._formatar(v_ij_new))
        print(f"{Colors.BOLD} ➔ Novos bias Ocultos (w0_new):{Colors.RESET}\n")

    # ==========================================
    # TESTES
    # ==========================================
    def log_iteracao_teste(self, indice_dado):
        print(f"\n{Colors.BOLD}{Colors.BLUE}{self.traco_grosso}")
        print(f" TESTANDO DADO #{indice_dado} ".center(60, "-"))
        print(f"{self.traco_grosso}{Colors.RESET}")

    def log_resultado_teste(self, previsto, esperado, saida_rede, erro):
        # Header sem os traços "---"
        print(f"\n{Colors.CYAN}[RESULTADO DO TESTE]{Colors.RESET}")

        # Cor vermelha/verde para erro/acerto
        cor_resultado = Colors.GREEN if str(previsto).casefold() == str(esperado).casefold() else Colors.RED

        print(f"{Colors.BOLD} ➔ Previsto: {cor_resultado}{previsto}{Colors.RESET}")
        print(f"{Colors.BOLD} ➔ Resposta: {esperado}{Colors.RESET}")
        print(f"{Colors.BOLD} ➔ Saída da Rede:{Colors.RESET}")
        print(self._formatar(saida_rede))
        print(f"{Colors.BOLD} ➔ Erro Total da Amostra:{Colors.RESET}")
        print(self._formatar(erro))