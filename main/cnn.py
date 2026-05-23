"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

Implementação de uma Rede Neural Convolucional (CNN) para classificação de imagens
do dataset Fashion MNIST.

Suporta dois modos de entrada:
  - 'bruto':   imagens 28x28 passadas diretamente para camadas Conv2D.
  - 'hog_lbp': vetores de características HOG+LBP passados para camadas Dense.

Suporta duas tarefas:
  - 'multiclasse': 10 classes do Fashion MNIST.
  - 'binaria':     2 classes (Camiseta=0 vs Calça=1).

Arquitetura CNN (modo bruto, com as images de verdade):

  Entrada: (28, 28, 1)
  Conv2D(32, kernel 3x3, ReLU) → BatchNorm → MaxPool(2x2)
  Conv2D(64, kernel 3x3, ReLU) → BatchNorm → MaxPool(2x2)
  Conv2D(128, kernel 3x3, ReLU) → BatchNorm
  Flatten → Dropout(0.4)
  Dense(256, ReLU) → Dropout(0.3)
  Dense(128, ReLU)
  Saída: Dense(n_classes, Softmax) ou Dense(1, Sigmoid)


Arquitetura DNN () (modo hog_lbp):

   Entrada: (D,) — vetor HOG+LBP
   Dense(256, ReLU) → Dropout(0.3)
   Dense(128, ReLU) → Dropout(0.2)
   Dense(64, ReLU)
   Saída: Dense(n_classes, Softmax) ou Dense(1, Sigmoid)


Escolhas de projeto:
  - ReLU nas camadas ocultas: evita o problema do gradiente que desaparece
    e converge mais rápido que sigmoid/tanh em CNNs.
  - BatchNormalization: normaliza as ativações por mini-batch, acelerando
    o treinamento e permitindo taxas de aprendizado maiores.
  - MaxPooling(2x2): reduz a dimensionalidade preservando as
    características mais expressivas (bordas, texturas).
  - Dropout: regularização que desativa neurônios aleatoriamente
    durante o treino, reduzindo overfitting.
  - Adam: otimizador que combina momentum e RMSProp
  - Softmax (multiclasse) / Sigmoid (binária): funções de ativação de saída
    adequadas para cada tipo de problema.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from logger_cnn import LoggerCNN
from writer_cnn import WriterCNN
from descritores import extrair_hog_lbp


# Nomes das 10 classes do Fashion MNIST (índice = rótulo numérico)
CLASSES_FASHION = [
    "Camiseta",   # 0 — T-shirt/top
    "Calça",      # 1 — Trouser
    "Pullover",   # 2
    "Vestido",    # 3 — Dress
    "Casaco",     # 4 — Coat
    "Sandália",   # 5 — Sandal
    "Camisa",     # 6 — Shirt
    "Tênis",      # 7 — Sneaker
    "Bolsa",      # 8 — Bag
    "Bota",       # 9 — Ankle boot
]

# Classes usadas na tarefa binária: Camiseta (0) vs Calça (1)
CLASSES_BINARIAS = {0: "Camiseta", 1: "Calça"}


class CNN:
    """
    Contém o processo completo da CNN:
      1. Construção do modelo (build)
      2. Treinamento (fit)
      3. Teste e métricas (teste)
      4. Escrita de resultados e pesos (via WriterCNN)
      5. Log de console (via LoggerCNN)
    """

    def __init__(
        self,
        tarefa: str = 'multiclasse',
        modo_dados: str = 'bruto',
        epocas: int = 15,
        batch_size: int = 64,
        taxa_aprendizado: float = 0.001,
        diretorio_saida: str = "../Saidas"
    ):
        """
        Args:
            tarefa:           'multiclasse' ou 'binaria'.
            modo_dados:       'bruto' (Conv2D) ou 'hog_lbp' (Dense com descritores).
            epocas:           número de épocas de treinamento.
            batch_size:       tamanho do mini-batch.
            taxa_aprendizado: taxa inicial do otimizador Adam.
            diretorio_saida:  pasta raiz para os arquivos de saída.
        """
        assert tarefa in ('multiclasse', 'binaria'), "tarefa deve ser 'multiclasse' ou 'binaria'"
        assert modo_dados in ('bruto', 'hog_lbp'), "modo_dados deve ser 'bruto' ou 'hog_lbp'"

        self.tarefa = tarefa
        self.modo_dados = modo_dados
        self.epocas = epocas
        self.batch_size = batch_size
        self.taxa_aprendizado = taxa_aprendizado

        # Número de neurônios de saída e função de ativação dependem da tarefa
        if tarefa == 'multiclasse':
            self.n_saida = 10
            self.ativacao_saida = 'softmax'
            self.funcao_perda = 'sparse_categorical_crossentropy'
            self.nomes_classes = CLASSES_FASHION
        else:
            # Binária: Camiseta (0) vs Calça (1) — saída escalar com sigmoid
            self.n_saida = 1
            self.ativacao_saida = 'sigmoid'
            self.funcao_perda = 'binary_crossentropy'
            self.nomes_classes = list(CLASSES_BINARIAS.values())

        self.model = None
        self.logger = LoggerCNN()
        self.writer = WriterCNN(diretorio_saida=diretorio_saida)

        # Configurações para log e persistência
        config = {
            "Épocas":              self.epocas,
            "Batch size":          self.batch_size,
            "Taxa de aprendizado": self.taxa_aprendizado,
            "Otimizador":          "Adam",
            "Função de perda":     self.funcao_perda,
            "Ativação saída":      self.ativacao_saida,
            "Nº classes":          self.n_saida if self.n_saida > 1 else 2,
            "Classes":             str(self.nomes_classes),
        }

        # Log no console e escrita em arquivo
        self.logger.log_configuracoes_cnn(self.tarefa, self.modo_dados, config)
        self.writer.write_hiperparametros_cnn(
            tarefa=self.tarefa,
            modo_dados=self.modo_dados,
            config=config
        )

    # ------------------------------------------------------------------
    # CONSTRUÇÃO DO MODELO
    # ------------------------------------------------------------------

    def _build_cnn_bruta(self, input_shape):
        """
        Constrói a CNN para dados brutos (imagens 28x28x1).

        Camadas convolucionais:
          - Conv2D com kernel 3x3: tamanho padrão (que equilibra campo receptivo
            e custo computacional) tirar essa parada aí. Não sei pelo que substituir.
            ReLU para não-linearidade.
          - BatchNormalization após cada Conv: estabiliza o treinamento.
          - MaxPooling 2x2: reduz altura e largura pela metade.

        Camadas densas:
          - Flatten: transforma o tensor 3D em vetor 1D.
          - Dropout(0.4): regularização mais agressiva antes da primeira Dense.
          - Dense(256) e Dense(128): capacidade suficiente para 10 classes.
        """
        modelo = keras.Sequential(name=f"CNN_{self.tarefa}_bruto")

        # --- Bloco convolucional 1 ---
        # 32 filtros: detecta características simples (bordas, texturas básicas)
        modelo.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                 input_shape=input_shape, padding='same',
                                 name='conv1'))
        modelo.add(layers.BatchNormalization(name='bn1'))
        # MaxPooling reduz 28x28 → 14x14
        modelo.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool1'))

        # --- Bloco convolucional 2 ---
        # 64 filtros: detecta padrões mais complexos (contornos, partes de objetos)
        modelo.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                                 padding='same', name='conv2'))
        modelo.add(layers.BatchNormalization(name='bn2'))
        # MaxPooling reduz 14x14 → 7x7
        modelo.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool2'))

        # --- Bloco convolucional 3 ---
        # 128 filtros: características de alto nível (formas globais das roupas)
        # Sem pooling: manter resolução 7x7 para não perder informação
        modelo.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu',
                                 padding='same', name='conv3'))
        modelo.add(layers.BatchNormalization(name='bn3'))

        # --- Camadas densas ---
        modelo.add(layers.Flatten(name='flatten'))
        modelo.add(layers.Dropout(0.4, name='dropout1'))
        modelo.add(layers.Dense(256, activation='relu', name='dense1'))
        modelo.add(layers.Dropout(0.3, name='dropout2'))
        modelo.add(layers.Dense(128, activation='relu', name='dense2'))

        # --- Camada de saída ---
        modelo.add(layers.Dense(self.n_saida, activation=self.ativacao_saida, name='saida'))

        return modelo

    def _build_dnn_descritores(self, input_dim: int):
        """
        Constrói uma DNN (rede totalmente conectada) para vetores HOG+LBP.

        Como os descritores já são representações compactas das imagens,
        não são necessárias camadas convolucionais.

        Args:
            input_dim: dimensão do vetor de entrada (D_hog + D_lbp).
        """
        modelo = keras.Sequential(name=f"DNN_{self.tarefa}_hog_lbp")

        modelo.add(layers.Input(shape=(input_dim,), name='entrada'))
        modelo.add(layers.Dense(256, activation='relu', name='dense1'))
        modelo.add(layers.Dropout(0.3, name='dropout1'))
        modelo.add(layers.Dense(128, activation='relu', name='dense2'))
        modelo.add(layers.Dropout(0.2, name='dropout2'))
        modelo.add(layers.Dense(64, activation='relu', name='dense3'))
        modelo.add(layers.Dense(self.n_saida, activation=self.ativacao_saida, name='saida'))

        return modelo

    def build(self, input_shape_ou_dim):
        """
        Constrói e compila o modelo conforme o modo de dados e a tarefa.

        Args:
            input_shape_ou_dim: tupla (H, W, C) para modo 'bruto',
                                 ou int D para modo 'hog_lbp'.
        """
        if self.modo_dados == 'bruto':
            self.model = self._build_cnn_bruta(input_shape=input_shape_ou_dim)
        else:
            self.model = self._build_dnn_descritores(input_dim=input_shape_ou_dim)

        # Adam: otimizador adaptativo padrão para CNNs
        otimizador = keras.optimizers.Adam(learning_rate=self.taxa_aprendizado)

        self.model.compile(
            optimizer=otimizador,
            loss=self.funcao_perda,
            metrics=['accuracy']
        )

        self.model.summary()

    # ------------------------------------------------------------------
    # TREINAMENTO
    # ------------------------------------------------------------------

    def fit(self, X_treino, y_treino, X_val=None, y_val=None):
        """
        Treina o modelo e guarda os pesos iniciais, finais e histórico.

        Usa um LoggerCallback para usar o LoggerCNN no final de cada época.

        Args:
            X_treino: array de entrada de treinamento.
            y_treino: array de rótulos de treinamento.
            X_val:    array de entrada de validação (opcional).
            y_val:    array de rótulos de validação (opcional).

        Returns:
            Objeto History do Keras.
        """
        n_val = len(X_val) if X_val is not None else int(len(X_treino) * 0.2)
        self.logger.log_inicio_treinamento(
            self.tarefa, self.modo_dados,
            n_treino=len(X_treino) - n_val,
            n_val=n_val,
            epocas=self.epocas
        )

        # Salva pesos ANTES do treinamento (inicializados aleatoriamente)
        self.writer.write_pesos_cnn(self.model, etapa="iniciais", tarefa=self.tarefa)

        dados_val = (X_val, y_val) if X_val is not None and y_val is not None else None

        # Referência ao logger para uso dentro do callback interno
        logger_ref = self.logger

        class LoggerCallback(keras.callbacks.Callback):
            """Ponte entre eventos do Keras e o LoggerCNN."""

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                logger_ref.log_resultado_epoca(
                    epoca=epoch,
                    loss=logs.get('loss', 0.0),
                    acc=logs.get('accuracy', 0.0),
                    val_loss=logs.get('val_loss'),
                    val_acc=logs.get('val_accuracy')
                )

            def on_train_end(self, logs=None):
                epocas_executadas = len(self.model.history.history.get('loss', []))
                if epocas_executadas < self.params.get('epochs', epocas_executadas):
                    logger_ref.log_parada_antecipada(epocas_executadas)

        callbacks = [
            LoggerCallback(),
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if dados_val else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=0  # silencia o output padrão; LoggerCNN assume o log
            )
        ]

        history = self.model.fit(
            X_treino, y_treino,
            epochs=self.epocas,
            batch_size=self.batch_size,
            validation_data=dados_val,
            validation_split=0.2 if dados_val is None else 0.0,
            callbacks=callbacks,
            verbose=0  # silencia o output padrão do Keras; LoggerCNN assume o log
        )

        # Salva pesos APÓS o treinamento
        self.writer.write_pesos_cnn(self.model, etapa="finais", tarefa=self.tarefa)

        # Salva histórico de loss/acurácia por época
        self.writer.write_historico_cnn(history, tarefa=self.tarefa)

        return history

    # ------------------------------------------------------------------
    # TESTE E MÉTRICAS
    # ------------------------------------------------------------------

    def teste(self, X_teste, y_teste):
        """
        Avalia o modelo no conjunto de teste e salva todos os artefatos.

        Para a tarefa binária, a saída da sigmoid é um escalar em [0,1]:
          - >= 0.5 → classe 1 (Calça)
          - <  0.5 → classe 0 (Camiseta)

        Para a tarefa multiclasse, argmax do vetor softmax determina a classe.

        Args:
            X_teste: array de entrada de teste.
            y_teste: array de rótulos verdadeiros (inteiros).

        Returns:
            dict com 'acuracia', 'matriz_confusao', 'y_pred'.
        """
        self.logger.log_inicio_teste_cnn(self.tarefa, self.modo_dados, len(X_teste))

        # Probabilidades brutas da rede
        y_proba = self.model.predict(X_teste, verbose=0)

        # Converte probabilidades em índices de classe
        if self.tarefa == 'binaria':
            # Sigmoid retorna shape (N, 1); achatamos para (N,)
            y_proba_flat = y_proba.flatten()
            y_pred = (y_proba_flat >= 0.5).astype(int)
            y_proba_para_csv = y_proba_flat
        else:
            y_pred = np.argmax(y_proba, axis=1)
            y_proba_para_csv = y_proba

        # Log individual de cada amostra de teste
        for i in range(len(y_teste)):
            esperado = self.nomes_classes[y_teste[i]]
            previsto = self.nomes_classes[y_pred[i]]
            proba_i = (y_proba_para_csv[i].tolist()
                       if hasattr(y_proba_para_csv[i], 'tolist')
                       else [float(y_proba_para_csv[i])])
            self.logger.log_resultado_teste_cnn(i, esperado, previsto, proba_i)

        # Calcula acurácia
        acertos = int(np.sum(y_pred == y_teste))
        total = len(y_teste)

        self.logger.log_acuracia_final(acertos, total, self.tarefa, self.modo_dados)

        # Monta matriz de confusão
        n = len(self.nomes_classes)
        matriz = [[0] * n for _ in range(n)]
        for real, prev in zip(y_teste, y_pred):
            matriz[int(real)][int(prev)] += 1

        self._print_matriz_confusao(matriz)

        # Persiste todos os artefatos de teste
        self.writer.write_saidas_teste_cnn(
            y_true=y_teste,
            y_pred=y_pred,
            y_pred_proba=y_proba_para_csv,
            nomes_classes=self.nomes_classes,
            tarefa=self.tarefa
        )
        self.writer.write_matriz_confusao_cnn(matriz, self.nomes_classes, tarefa=self.tarefa)
        self.writer.write_acuracia_cnn(acertos, total, tarefa=self.tarefa)

        return {
            'acuracia': acertos / total,
            'matriz_confusao': matriz,
            'y_pred': y_pred
        }

    def _print_matriz_confusao(self, matriz):
        """
        Imprime a matriz de confusão no console.
        Reutiliza traco_grosso e traco_fino herdados pelo LoggerCNN.
        """
        n = len(self.nomes_classes)
        largura = max(len(c) for c in self.nomes_classes) + 2

        print(f"\n{self.logger.traco_grosso}")
        print(f"MATRIZ DE CONFUSÃO — {self.tarefa.upper()}".center(60))
        print(self.logger.traco_grosso)

        cabecalho = " " * (largura + 3) + "".join(c[:4].center(6) for c in self.nomes_classes)
        print(cabecalho)
        print("-" * (6 * n + largura + 3))

        for i in range(n):
            linha = (self.nomes_classes[i][:largura]).ljust(largura) + " | "
            linha += "".join(str(matriz[i][j]).center(6) for j in range(n))
            print(linha)

        print("-" * 60)
