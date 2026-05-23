"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054
"""

import os
import csv
import numpy as np
import sys
import csv as _csv

# Garante que o pacote loggers seja encontrado independente de onde o script é chamado
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from writer import Writer


class WriterCNN(Writer):
    """
    Subclasse de Writer com métodos adicionais voltados para a CNN.

    Os callbacks nativos do Keras (CSVLogger, ModelCheckpoint) escrevem
    diretamente nos arquivos; os métodos abaixo apenas fornecem os caminhos
    e fazem o pós-processamento mínimo para manter os formatos.
    """

    def __init__(self, diretorio_saida="../Saidas", prefixo="cnn"):
        super().__init__(diretorio_saida=diretorio_saida)

        pasta_nova = os.path.join(
            diretorio_saida,
            f"{prefixo}_{os.path.basename(self.pasta_atual)}"
        )
        os.rename(self.pasta_atual, pasta_nova)
        self.pasta_atual = pasta_nova

    # ==========================================
    # ARQUIVO 1: HIPERPARÂMETROS CNN
    # ==========================================
    def write_hiperparametros_cnn(self, tarefa, modo_dados, config):
        nome = f"1_hiperparametros_{tarefa}.txt"
        caminho = self._obter_caminho(nome)
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(f"=== HIPERPARÂMETROS DA CNN ({tarefa.upper()}) ===\n")
            f.write(f"Modo de dados:   {modo_dados}\n")
            for chave, valor in config.items():
                f.write(f"{chave:<25}{valor}\n")

    # ==========================================
    # CAMINHOS PARA CALLBACKS DO KERAS
    # ==========================================
    def caminho_pesos_h5(self, etapa, tarefa):
        num = "2" if etapa == "iniciais" else "3"
        return self._obter_caminho(f"{num}_pesos_{etapa}_{tarefa}.weights.h5")

    def caminho_historico_nativo(self, tarefa):
        return self._obter_caminho(f"4_historico_treinamento_{tarefa}_keras.csv")

    # ==========================================
    # ARQUIVO 2 E 3: PESOS INICIAIS E FINAIS
    # ==========================================
    def write_pesos_cnn(self, model, etapa, tarefa):
        self.write_resumo_pesos_cnn(model, etapa=etapa, tarefa=tarefa)
        model.save_weights(self.caminho_pesos_h5(etapa=etapa, tarefa=tarefa))

    def write_resumo_pesos_cnn(self, model, etapa, tarefa):
        num = "2" if etapa == "iniciais" else "3"
        caminho_txt = self._obter_caminho(f"{num}_pesos_{etapa}_{tarefa}.txt")
        with open(caminho_txt, 'w', encoding='utf-8') as f:
            f.write(f"=== PESOS {etapa.upper()} — CNN ({tarefa.upper()}) ===\n\n")
            for layer in model.layers:
                pesos = layer.get_weights()
                if not pesos:
                    continue
                f.write(f"Camada: {layer.name}\n")
                for idx, p in enumerate(pesos):
                    f.write(f"  Tensor {idx}: shape={p.shape}  "
                            f"min={np.min(p):.6f}  max={np.max(p):.6f}  "
                            f"mean={np.mean(p):.6f}  std={np.std(p):.6f}\n")
                f.write("\n")

    # ==========================================
    # ARQUIVO 4: HISTÓRICO DE ERRO POR ÉPOCA
    # ==========================================
    def finalizar_historico_cnn(self, caminho_nativo, tarefa):
        caminho_final = self._obter_caminho(f"4_historico_treinamento_{tarefa}.csv")

        with open(caminho_nativo, 'r', encoding='utf-8') as f:
            linhas = list(csv.reader(f))

        cabecalho, dados = linhas[0], linhas[1:]
        cabecalho = ["Epoca"] + cabecalho[1:]

        with open(caminho_final, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(cabecalho)
            for linha in dados:
                if not linha:          # ignora linhas em branco do CSVLogger
                    continue
                w.writerow([int(linha[0]) + 1] + linha[1:])

        if os.path.exists(caminho_nativo):
            os.remove(caminho_nativo)

    # ==========================================
    # ARQUIVO 5: SAÍDAS DO TESTE
    # ==========================================
    def write_saidas_teste_cnn(self, y_true, y_pred, y_pred_proba, nomes_classes, tarefa):
        caminho = self._obter_caminho(f"5_saidas_teste_{tarefa}.csv")
        with open(caminho, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Esperado", "Previsto", "Acertou", "Vetor_Saida_Rede"])
            for i in range(len(y_true)):
                esperado = nomes_classes[y_true[i]]
                previsto = nomes_classes[y_pred[i]]
                acertou = "Sim" if y_true[i] == y_pred[i] else "Nao"
                proba = y_pred_proba[i]
                vetor_str = self._formatar_completo(
                    proba.tolist() if hasattr(proba, 'tolist') else [float(proba)]
                )
                writer.writerow([esperado, previsto, acertou, vetor_str])

    # ==========================================
    # ARQUIVO 6: MATRIZ DE CONFUSÃO
    # ==========================================
    def write_matriz_confusao_cnn(self, matriz, nomes_classes, tarefa):
        caminho = self._obter_caminho(f"6_matriz_confusao_{tarefa}.csv")
        with open(caminho, 'w', newline='', encoding='utf-8') as f:
            w = _csv.writer(f)
            w.writerow([""] + nomes_classes)
            for i, nome in enumerate(nomes_classes):
                w.writerow([nome] + list(matriz[i]))

    # ==========================================
    # ARQUIVO 7: ACURÁCIA
    # ==========================================
    def write_acuracia_cnn(self, count, total, tarefa):
        caminho = self._obter_caminho(f"7_acuracia_{tarefa}.txt")
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(f"=== ACURÁCIA DO TESTE ({tarefa.upper()}) ===\n")
            f.write(f"Acertos:  {count}/{total}\n")
            f.write(f"Acurácia: {round(count / total * 100, 2)}%\n")