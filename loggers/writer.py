import numpy as np
import os
import csv
from datetime import datetime


class Writer:
    def __init__(self, diretorio_saida="../Saidas"):
        """
        Cria uma pasta única para a execução atual dentro de 'Saidas'.
        """
        self.diretorio_saida = diretorio_saida
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pasta_atual = os.path.join(self.diretorio_saida, f"execucao_{self.timestamp}")

        if not os.path.exists(self.pasta_atual):
            os.makedirs(self.pasta_atual)

    def _obter_caminho(self, nome_arquivo):
        return os.path.join(self.pasta_atual, nome_arquivo)

    def _formatar_completo(self, dado):
        """Transforma matrizes do Numpy em strings completas, sem omissão (...)"""
        if hasattr(dado, 'tolist'):
            dado = dado.tolist()
        if isinstance(dado, list):
            if len(dado) > 0 and isinstance(dado[0], list):
                linhas = [str(np.round(np.array(linha, dtype=float), 6).tolist()) for linha in dado]
                return "\n".join(linhas)
            return str(np.round(np.array(dado, dtype=float), 6).tolist())
        return str(round(dado, 6) if isinstance(dado, float) else dado)


    # ==========================================
    # ARQUIVO 1: HIPERPARÂMETROS
    # ==========================================
    def write_hiperparametros(self, n_entradas, n_ocultas, n_saidas, taxa_aprendizado, ativacao, epocas, limiar_erro):
        caminho = self._obter_caminho("1_hiperparametros.txt")
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write("=== HIPERPARAMETROS DA REDE ===\n")
            f.write(f"Arquitetura:         {n_entradas} (Entrada) -> {n_ocultas} (Oculta) -> {n_saidas} (Saída)\n")
            f.write(f"Taxa de Aprendizado: {taxa_aprendizado}\n")
            f.write(f"Função de Ativação:  {ativacao}\n")
            f.write(f"Épocas:              {epocas}\n")
            f.write(f"Limiar de Erro:      {limiar_erro if limiar_erro is not None else 'inativo'}\n")

    # ==========================================
    # ARQUIVOS 2 E 3: PESOS INICIAIS E FINAIS
    # ==========================================
    def write_pesos(self, W, w0, B, b0, etapa="iniciais"):
        caminho = self._obter_caminho(f"2_pesos_{etapa}.txt" if etapa == "iniciais" else "3_pesos_finais.txt")
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write("=== PESOS ENTRADA -> OCULTA (W) ===\n")
            f.write(self._formatar_completo(W) + "\n\n")
            f.write("=== BIAS DA CAMADA OCULTA (w0) ===\n")
            f.write(self._formatar_completo(w0) + "\n\n")
            f.write("=== PESOS OCULTA -> SAÍDA (B) ===\n")
            f.write(self._formatar_completo(B) + "\n\n")
            f.write("=== BIAS DA CAMADA DE SAÍDA (b0) ===\n")
            f.write(self._formatar_completo(b0) + "\n")

    # ==========================================
    # ARQUIVO 4: ERROS DO TREINAMENTO
    # ==========================================
    def write_erros(self, erros_por_epoca, erros_por_iteracao):
        caminho_epoca = self._obter_caminho("4_erros_treinamento_epoca.csv")
        with open(caminho_epoca, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoca", "Erro_Medio_da_Epoca"])
            for epoca, erro in enumerate(erros_por_epoca):
                writer.writerow([epoca + 1, erro])

        caminho_iter = self._obter_caminho("4_erros_treinamento_iteracao.csv")
        with open(caminho_iter, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoca", "Iteracao", "Erro_Quadratico"])
            for entrada in erros_por_iteracao:
                writer.writerow([entrada['epoca'], entrada['iteracao'], entrada['erro']])

    # ==========================================
    # ARQUIVO 5: SAÍDAS DO TESTE
    # ==========================================
    def write_saidas_teste(self, resultados):
        caminho = self._obter_caminho("5_saidas_teste.csv")
        with open(caminho, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Letra_Esperada", "Letra_Prevista", "Acertou", "Erro_Total_Amostra", "Vetor_Saida_Rede"])
            for res in resultados:
                acertou = "Sim" if str(res['esperado']).casefold() == str(res['previsto']).casefold() else "Nao"
                writer.writerow([
                    res['esperado'],
                    res['previsto'],
                    acertou,
                    res['erro'],
                    self._formatar_completo(res['saida'])
                ])

    # ==========================================
    # ARQUIVO 6: MATRIZ DE CONFUSÃO
    # ==========================================
    def write_matriz_confusao(self, matriz, letras):
        caminho = self._obter_caminho("6_matriz_confusao.csv")
        with open(caminho, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Construção do cabeçalho
            writer.writerow([""] + letras)
            for i, letra in enumerate(letras):
                writer.writerow([letra] + matriz[i])