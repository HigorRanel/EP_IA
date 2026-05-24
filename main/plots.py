"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

Módulo de plotagem (matplotlib) para os gráficos exigidos no vídeo:
  - Curva de comportamento do erro (EQM) por época: treino vs. validação
    -> evidencia a convergência e o ponto de parada antecipada (slides 10/11).
  - Matriz de confusão como heatmap (slide 12).

As funções salvam a figura em arquivo PNG e, opcionalmente, exibem na tela.
Usa backend não-interativo quando apenas salva, evitando erros em ambientes
sem display (servidores, execução headless).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend não-interativo: salva PNG sem precisar de display
import matplotlib.pyplot as plt


def plotar_curva_erro(erros_treino, erros_val=None, caminho_saida=None,
                      titulo="Comportamento do Erro (EQM) por Época",
                      mostrar=False):
    """
    Plota o EQM de treino (e de validação, se houver) ao longo das épocas.

    Args:
        erros_treino: lista do EQM de treino por época.
        erros_val:    lista do EQM de validação por época (ou None).
        caminho_saida: caminho do PNG a salvar (ou None para não salvar).
        titulo:       título do gráfico.
        mostrar:      se True, exibe a janela (plt.show()).
    """
    epocas = range(1, len(erros_treino) + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epocas, erros_treino, label="Treino", color="#1f77b4", linewidth=2)

    if erros_val is not None and len(erros_val) > 0:
        ax.plot(range(1, len(erros_val) + 1), erros_val,
                label="Validação", color="#d62728", linewidth=2)

        # Marca a época de menor erro de validação (melhor modelo)
        melhor_epoca = int(np.argmin(erros_val)) + 1
        melhor_val = erros_val[melhor_epoca - 1]
        ax.axvline(melhor_epoca, color="gray", linestyle="--", linewidth=1)
        ax.scatter([melhor_epoca], [melhor_val], color="#d62728", zorder=5)
        ax.annotate(f"melhor val\n(época {melhor_epoca})",
                    xy=(melhor_epoca, melhor_val),
                    xytext=(10, 15), textcoords="offset points",
                    fontsize=9, color="gray")

    ax.set_xlabel("Época")
    ax.set_ylabel("EQM (Erro Quadrático Médio)")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if caminho_saida:
        fig.savefig(caminho_saida, dpi=130)
        print(f" Gráfico de erro salvo em: {caminho_saida}")
    if mostrar:
        plt.show()
    plt.close(fig)


def plotar_matriz_confusao(matriz, letras, caminho_saida=None,
                           titulo="Matriz de Confusão", mostrar=False):
    """
    Plota a matriz de confusão como heatmap, com os valores anotados.

    Args:
        matriz:        matriz NxN (lista de listas ou array) de contagens.
        letras:        rótulos das classes (eixos).
        caminho_saida: caminho do PNG a salvar (ou None).
        titulo:        título do gráfico.
        mostrar:       se True, exibe a janela.
    """
    matriz = np.array(matriz)
    n = len(letras)

    # Tamanho da figura cresce com o número de classes (26 letras -> grande)
    lado = max(6, n * 0.45)
    fig, ax = plt.subplots(figsize=(lado, lado))

    im = ax.imshow(matriz, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Nº de amostras")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(letras, fontsize=8)
    ax.set_yticklabels(letras, fontsize=8)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Esperado")
    ax.set_title(titulo)

    # Anota cada célula com sua contagem (texto escuro/claro conforme o fundo)
    limiar = matriz.max() / 2.0 if matriz.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            v = matriz[i, j]
            if v > 0:  # não polui células zeradas
                ax.text(j, i, str(int(v)), ha="center", va="center",
                        fontsize=7,
                        color="white" if v > limiar else "black")

    fig.tight_layout()

    if caminho_saida:
        fig.savefig(caminho_saida, dpi=130)
        print(f" Matriz de confusão salva em: {caminho_saida}")
    if mostrar:
        plt.show()
    plt.close(fig)