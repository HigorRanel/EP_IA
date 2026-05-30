"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

Módulo para criação de arquivos/saídas do projeto
"""

import numpy as np
import os
import csv
from datetime import datetime


class Writer:
    def __init__(self, diretorio_saida="./Saidas"):
        """
        Cria uma pasta única para a execução atual dentro de 'Saidas'

        Recebe como parâmetros:
        1) diretorio_saida: diretório-raiz onde a pasta da execução será criada

        Retorna: None (cria a pasta de saída)
        """
        self.diretorio_saida = diretorio_saida
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pasta_atual = os.path.join(self.diretorio_saida, f"execucao_{self.timestamp}")

        if not os.path.exists(self.pasta_atual):
            os.makedirs(self.pasta_atual)

    def obter_caminho(self, nome_arquivo):
        return os.path.join(self.pasta_atual, nome_arquivo)

    def formatar_completo(self, dado):
        """Transforma matrizes do Numpy em strings completas"""
        if hasattr(dado, 'tolist'):
            dado = dado.tolist()
        if isinstance(dado, list):
            if len(dado) > 0 and isinstance(dado[0], list):
                linhas = [str(np.round(np.array(linha, dtype=float), 6).tolist()) for linha in dado]
                return "\n".join(linhas)
            return str(np.round(np.array(dado, dtype=float), 6).tolist())
        return str(round(dado, 6) if isinstance(dado, float) else dado)


    def write_hiperparametros(self, n_entradas, n_ocultas, n_saidas, taxa_aprendizado, ativacao, epocas):
        """
        Escreve os hiperparâmetros em um arquivo chamado
        '1_hiperparametros_multiclasse.txt':
            1. Arquitetura: número de neurônios na camada de entrada, oculta e de saída
            2. Taxa de aprendizado
            3. Função de ativação
            4. Número de épocas
            5. Limiar de erro

        Recebe como parâmetros:
        1) n_entradas: nº de neurônios da camada de entrada
        2) n_ocultas: nº de neurônios da camada oculta
        3) n_saidas: nº de neurônios da camada de saída
        4) taxa_aprendizado: taxa de aprendizado usada no treino
        5) ativacao: nome da função de ativação
        6) epocas: nº de épocas configurado

        Retorna: None (grava o arquivo de hiperparâmetros)
        """
        caminho = self.obter_caminho("1_hiperparametros.txt")
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write("HIPERPARAMETROS DA REDE\n")
            f.write(f"Arquitetura: {n_entradas} (Entrada): {n_ocultas} (Oculta): {n_saidas} (Saída)\n")
            f.write(f"Taxa de Aprendizado: {taxa_aprendizado}\n")
            f.write(f"Função de Ativação: {ativacao}\n")
            f.write(f"Épocas: {epocas}\n")

    def write_pesos(self, W, w0, B, b0, etapa="iniciais"):
        """
        Escreve os pesos de cada camada (entrada, oculta e de saída) na etapa inicial,
        e também quando terminamos o treinamento

        Recebe como parâmetros:
        1) W: matriz de pesos entrada - oculta
        2) w0: vetor de bias da camada oculta
        3) B: matriz de pesos oculta - saída
        4) b0: vetor de bias da camada de saída
        5) etapa: "iniciais" ou "finais", define o nome do arquivo gerado

        Retorna: None (grava o arquivo de pesos)
        """
        caminho = self.obter_caminho(f"2_pesos_{etapa}.txt" if etapa == "iniciais" else "3_pesos_finais.txt")
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write("PESOS ENTRADA OCULTA (W)\n")
            f.write(self.formatar_completo(W) + "\n\n")
            f.write("BIAS DA CAMADA OCULTA (w0)\n")
            f.write(self.formatar_completo(w0) + "\n\n")
            f.write("PESOS OCULTA SAÍDA (B)\n")
            f.write(self.formatar_completo(B) + "\n\n")
            f.write("BIAS DA CAMADA DE SAÍDA (b0)\n")
            f.write(self.formatar_completo(b0) + "\n")

    def write_erros(self, erros_por_epoca, erros_por_iteracao, erros_val_por_epoca=None):
        """
        Escreve o erro de treino e de validação para cada época em um arquivo chamado
        '4_erros_treinamento_epoca.csv' e também
        escreve o erro quadrático de cada iteração para
        todas as épocas em um arquivo chamado '4_erros_treinamento_iteracao.csv'

        Recebe como parâmetros:
        1) erros_por_epoca: lista com o EQM de treino de cada época
        2) erros_por_iteracao: lista de dicionários {epoca, iteracao, erro}
        de cada iteração
        3) erros_val_por_epoca: lista com o EQM de validação de cada época

        Retorna: None (grava os dois arquivos CSV de erros)
        """
        tem_val = erros_val_por_epoca is not None and len(erros_val_por_epoca) > 0

        caminho_epoca = self.obter_caminho("4_erros_treinamento_epoca.csv")
        with open(caminho_epoca, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if tem_val:
                writer.writerow(["Epoca", "Erro_Treino", "Erro_Validacao"])
            else:
                writer.writerow(["Epoca", "Erro_Treino"])
            for epoca, erro in enumerate(erros_por_epoca):
                if tem_val:
                    # Protege contra tamanhos diferentes (a parada antecipada
                    # interrompe ambas as listas na mesma época, mas por segurança)
                    val = erros_val_por_epoca[epoca] if epoca < len(erros_val_por_epoca) else ""
                    writer.writerow([epoca + 1, erro, val])
                else:
                    writer.writerow([epoca + 1, erro])

        caminho_iter = self.obter_caminho("4_erros_treinamento_iteracao.csv")
        with open(caminho_iter, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoca", "Iteracao", "Erro_Quadratico"])
            for entrada in erros_por_iteracao:
                writer.writerow([entrada['epoca'], entrada['iteracao'], entrada['erro']])

    def write_saidas_teste(self, resultados):
        """
        Escreve para cada dado de teste a letra esperada, a letra prevista pelo modelo,
        se o modelo acertou ou não, o erro total da amostra e também o vetor de saída
        da rede (array de tamanho 26 que indica o valor de saída de cada
        neurônio da camada de saída)

        Recebe como parâmetros:
        1) resultados: lista de dicionários (um por amostra de teste) com esperado,
        previsto, saida e erro

        Retorna: None (grava o arquivo '5_saidas_teste.csv')
        """
        caminho = self.obter_caminho("5_saidas_teste.csv")
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
                    self.formatar_completo(res['saida'])
                ])

    def write_matriz_confusao(self, matriz, letras):
        """
        Escreve a matriz de confusão em um arquivo chamado '6_matriz_confusão.csv'

        Recebe como parâmetros:
        1) matriz: matriz de confusão,
            linha = esperado
            coluna = previsto
        2) letras
        """
        caminho = self.obter_caminho("6_matriz_confusao.csv")
        with open(caminho, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Construção do cabeçalho
            writer.writerow([""] + letras)
            for i, letra in enumerate(letras):
                writer.writerow([letra] + matriz[i])

    def write_acuracia(self, count, total):
        """
        Escreve o número de acertos totais da rede também a sua acurácia em um arquivo chamado
        '7_acuracia.txt'

        Recebe como parâmetros:
        1) count: nº de amostras de teste classificadas corretamente
        2) total: nº total de amostras de teste
        """
        caminho = self.obter_caminho("7_acuracia.txt")
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write("ACURÁCIA DO TESTE\n")
            f.write(f"Acertos:  {count}/{total}\n")
            f.write(f"Acurácia: {round(count / total * 100, 2)}%\n")