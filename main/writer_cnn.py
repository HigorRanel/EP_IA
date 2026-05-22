"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

WriterCNN: extensão do Writer base para persistência de artefatos da CNN.

Reutiliza toda a infraestrutura de diretórios e formatação do Writer original,
adicionando métodos específicos para os artefatos exigidos pela CNN:
- Hiperparâmetros da arquitetura CNN
- Pesos iniciais e finais (via H5/arquivo de texto resumido)
- Histórico de erro por época
- Saídas do teste (multiclasse e binário)
- Matriz de confusão
- Acurácia
"""

import os
import csv
import numpy as np
import sys

# Garante que o pacote loggers seja encontrado independente de onde o script é chamado
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from writer import Writer


class WriterCNN(Writer):
    """
    Subclasse de Writer com métodos adicionais voltados para a CNN.

    Herda:
    - __init__: criação da pasta de execução com timestamp
    - _obter_caminho: montagem de caminhos dentro da pasta de execução
    - _formatar_completo: formatação de arrays NumPy para string

    Adiciona:
    - write_hiperparametros_cnn
    - write_pesos_cnn (resumo textual + caminho do arquivo .weights.h5)
    - write_historico_cnn
    - write_saidas_teste_cnn
    - write_matriz_confusao_cnn  (reusa write_matriz_confusao do pai)
    - write_acuracia_cnn         (reusa write_acuracia do pai, com prefixo de arquivo)
    """

    def __init__(self, diretorio_saida="../Saidas", prefixo="cnn"):
        """
        Args:
            diretorio_saida: pasta raiz onde as subpastas de execução são criadas.
            prefixo: prefixo do nome da pasta para distinguir execuções CNN de MLP.
        """
        # Chama o __init__ do pai para criar self.pasta_atual com timestamp
        super().__init__(diretorio_saida=diretorio_saida)

        # Renomeia a pasta para incluir o prefixo "cnn_" (o pai cria "execucao_<ts>")
        pasta_nova = os.path.join(
            diretorio_saida,
            f"{prefixo}_{os.path.basename(self.pasta_atual)}"
        )
        os.rename(self.pasta_atual, pasta_nova)
        self.pasta_atual = pasta_nova

    # ==========================================
    # ARQUIVO 1: HIPERPARÂMETROS CNN
    # ==========================================
    def write_hiperparametros_cnn(self, tarefa: str, modo_dados: str, config: dict):
        """
        Persiste os hiperparâmetros da arquitetura e treinamento da CNN.

        Args:
            tarefa: 'multiclasse' ou 'binaria'.
            modo_dados: 'bruto', 'hog_lbp', etc.
            config: dicionário com os hiperparâmetros (épocas, batch_size, lr, etc.).
        """
        nome = f"1_hiperparametros_{tarefa}.txt"
        caminho = self._obter_caminho(nome)
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(f"=== HIPERPARÂMETROS DA CNN ({tarefa.upper()}) ===\n")
            f.write(f"Modo de dados:   {modo_dados}\n")
            for chave, valor in config.items():
                f.write(f"{chave:<25}{valor}\n")

    # ==========================================
    # ARQUIVO 2 E 3: PESOS INICIAIS E FINAIS
    # ==========================================
    def write_pesos_cnn(self, model, etapa: str, tarefa: str):
        """
        Salva um resumo textual dos pesos (shape e estatísticas por camada)
        e o arquivo de pesos completo em formato .weights.h5 do Keras.

        O arquivo .weights.h5 contém os pesos completos e pode ser carregado
        via model.load_weights(). O resumo .txt é para leitura humana.

        Args:
            model: modelo Keras já compilado.
            etapa: 'iniciais' ou 'finais'.
            tarefa: 'multiclasse' ou 'binaria'.
        """
        num = "2" if etapa == "iniciais" else "3"
        # Resumo textual
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

        # Pesos completos em H5
        caminho_h5 = self._obter_caminho(f"{num}_pesos_{etapa}_{tarefa}.weights.h5")
        model.save_weights(caminho_h5)

    # ==========================================
    # ARQUIVO 4: HISTÓRICO DE ERRO POR ÉPOCA
    # ==========================================
    def write_historico_cnn(self, history, tarefa: str):
        """
        Persiste o histórico de loss e acurácia por época retornado pelo Keras.

        Args:
            history: objeto History retornado por model.fit().
            tarefa: 'multiclasse' ou 'binaria'.
        """
        caminho = self._obter_caminho(f"4_historico_treinamento_{tarefa}.csv")
        with open(caminho, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            colunas = list(history.history.keys())
            writer.writerow(["Epoca"] + colunas)
            n_epocas = len(history.history[colunas[0]])
            for epoca in range(n_epocas):
                linha = [epoca + 1] + [history.history[col][epoca] for col in colunas]
                writer.writerow(linha)

    # ==========================================
    # ARQUIVO 5: SAÍDAS DO TESTE
    # ==========================================
    def write_saidas_teste_cnn(self, y_true, y_pred, y_pred_proba, nomes_classes: list, tarefa: str):
        """
        Persiste as saídas do teste: classe esperada, classe prevista, acerto e
        o vetor completo de probabilidades produzido pela softmax/sigmoid.

        Args:
            y_true: array de inteiros com os rótulos verdadeiros.
            y_pred: array de inteiros com os rótulos previstos.
            y_pred_proba: array (N, C) ou (N,) com as probabilidades de saída.
            nomes_classes: lista de strings com o nome de cada classe.
            tarefa: 'multiclasse' ou 'binaria'.
        """
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
    def write_matriz_confusao_cnn(self, matriz, nomes_classes: list, tarefa: str):
        """
        Persiste a matriz de confusão, reutilizando o formato do Writer pai
        mas com nome de arquivo separado por tarefa.

        Args:
            matriz: lista de listas (N x N) com contagens.
            nomes_classes: rótulos das linhas/colunas.
            tarefa: 'multiclasse' ou 'binaria'.
        """
        caminho = self._obter_caminho(f"6_matriz_confusao_{tarefa}.csv")
        import csv as _csv
        with open(caminho, 'w', newline='', encoding='utf-8') as f:
            w = _csv.writer(f)
            w.writerow([""] + nomes_classes)
            for i, nome in enumerate(nomes_classes):
                w.writerow([nome] + list(matriz[i]))

    # ==========================================
    # ARQUIVO 7: ACURÁCIA
    # ==========================================
    def write_acuracia_cnn(self, count: int, total: int, tarefa: str):
        """
        Persiste a acurácia final do teste.

        Args:
            count: número de acertos.
            total: total de amostras testadas.
            tarefa: 'multiclasse' ou 'binaria'.
        """
        caminho = self._obter_caminho(f"7_acuracia_{tarefa}.txt")
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(f"=== ACURÁCIA DO TESTE ({tarefa.upper()}) ===\n")
            f.write(f"Acertos:  {count}/{total}\n")
            f.write(f"Acurácia: {round(count / total * 100, 2)}%\n")
