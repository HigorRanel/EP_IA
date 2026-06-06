"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

Treina e testa a MLP na porta lógica AND, para testar se as formulas utilizadas estão corretas.

Diferente do main.py (letras): aqui a rede tem 1 saída só, lida por limiar (0.5)
em vez de argmax. E como a sigmoide vive em (0, 1), o rótulo bipolar (-1/1) vira
alvo 0/1 (1 = verdadeiro, -1 = falso).
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp import MLP
import loggers.writer as writer_mod


def carregar_porta():
    """Define diretamente a amostra (1, 1) -> rótulo 1 (verdadeiro na AND).
    Devolve (X, alvo, rotulo_original) no mesmo formato esperado pelo restante
    do código."""
    X = pd.DataFrame([[1.0, 1.0]], columns=[0, 1])
    rotulo_original = np.array([1])
    alvo = np.array([[1.0]])

    return X, alvo, rotulo_original


def testar_limiar(mlp, X, rotulo_original, limiar=0.5):
    """Roda o feedforward de cada amostra e decide por limiar: saída >= limiar é
    verdadeiro (1), senão falso (-1). Devolve (acertos, total, linhas), com as
    linhas detalhando cada amostra."""
    acertos = 0
    linhas = []

    for i in range(X.shape[0]):
        mlp.forward(X.iloc[i])
        saida = float(np.array(mlp.y)[0])
        previsto = 1 if saida >= limiar else -1
        esperado = int(rotulo_original[i])
        correto = (previsto == esperado)
        acertos += int(correto)

        linhas.append({
            'x1': float(X.iat[i, 0]),
            'x2': float(X.iat[i, 1]),
            'esperado': esperado,
            'saida_rede': round(saida, 4),
            'previsto': previsto,
            'acertou': 'Sim' if correto else 'Nao',
        })

    return acertos, X.shape[0], linhas


def matriz_confusao_binaria(linhas):
    """Matriz de confusão 2x2 (linha = esperado, coluna = previsto), classes na
    ordem [-1, 1] (falso, verdadeiro)."""
    classes = [-1, 1]
    indice = {c: k for k, c in enumerate(classes)}
    matriz = [[0, 0], [0, 0]]

    for linha in linhas:
        i = indice[linha['esperado']]
        j = indice[linha['previsto']]
        matriz[i][j] += 1

    return matriz


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    X, alvo, rotulo_original = carregar_porta()

    # Os arquivos da execução (pesos, erros, gráfico) são gravados pelo Writer
    # interno da MLP. Direcionamos a pasta desta execução de teste para
    # Saidas_Teste_De_Mesa/execucao_teste_<anomesdia_horaminseg>.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta_teste = os.path.join(BASE_DIR, 'Saidas_Teste_De_Mesa', f'execucao_teste_{timestamp}')

    def _writer_pasta_fixa(self, diretorio_saida="./Saidas_Teste_De_Mesa"):
        self.diretorio_saida = diretorio_saida
        self.pasta_atual = pasta_teste
        os.makedirs(self.pasta_atual, exist_ok=True)

    writer_mod.Writer.__init__ = _writer_pasta_fixa

    np.random.seed(42)  # reprodutibilidade dos pesos sorteados

    # Rede ajustada: 2 entradas, 2 ocultos (de sobra p/ a AND) e 1 saída.
    # Dá p/ aumentar os ocultos que o resultado continua certo.
    mlp = MLP(
        comprimento_entrada=2,
        comprimento_oculta=2,
        comprimento_saida=1,
        epocas=1,
        taxa_de_aprendizado=0.5,
        ini_pesos='teste_mesa',
        ini_bias='teste_mesa',
    )

    print("\nConjunto AND:")
    for i in range(X.shape[0]):
        print(f"  x1={X.iat[i, 0]:>2.0f}  x2={X.iat[i, 1]:>2.0f}  "
              f"-> rótulo {int(rotulo_original[i]):>2d}  (alvo {alvo[i, 0]:.0f})")

    # Sem validação: são só 4 amostras, todas viram treino e teste.
    mlp.fit(X, alvo)

    acertos, total, linhas = testar_limiar(mlp, X, rotulo_original)

    print("\n" + "=" * 60)
    print("TESTE DA PORTA AND (leitura por limiar = 0.5)".center(60))
    print("=" * 60)
    print(f"{'x1':>4} {'x2':>4} {'esperado':>10} {'saída':>9} {'previsto':>10} {'ok':>5}")
    print("-" * 60)
    for l in linhas:
        print(f"{l['x1']:>4.0f} {l['x2']:>4.0f} {l['esperado']:>10d} "
              f"{l['saida_rede']:>9.4f} {l['previsto']:>10d} {l['acertou']:>5}")

    print("-" * 60)
    print(f"Acurácia: {acertos}/{total} = {round(acertos / total * 100, 2)}%")

    matriz = matriz_confusao_binaria(linhas)
    print("\nMATRIZ DE CONFUSÃO (linha = esperado, coluna = previsto)")
    print(f"{'':>8}{'prev -1':>9}{'prev 1':>9}")
    print(f"{'real -1':>8}{matriz[0][0]:>9}{matriz[0][1]:>9}")
    print(f"{'real  1':>8}{matriz[1][0]:>9}{matriz[1][1]:>9}")
    print(f"\nArquivos da execução (pesos, erros, gráfico) salvos em "
          f"./Saidas_Teste_De_Mesa/execucao_teste_{timestamp}\n")


if __name__ == '__main__':
    main()
