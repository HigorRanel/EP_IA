"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054
"""

import numpy as np
from skimage.feature import hog, local_binary_pattern


def extrair_hog(imagens):
    """
    Extrai descritores HOG de um conjunto de imagens.

    Parâmetros do HOG escolhidos:
    - orientations=8: número de bins do histograma de gradientes por célula.
      Valor padrão amplamente usado; suficiente para capturar formas de roupas.
    - pixels_per_cell=(4, 4): tamanho de cada célula local. Células menores
      capturam detalhes mais finos; 4x4 é adequado para imagens 28x28.
    - cells_per_block=(2, 2): normalização ocorre em blocos de 2x2 células,
      tornando o descritor mais robusto a variações de contraste.
    """
    descritores = []
    for img in imagens:
        # skimage espera imagens com valores float em [0,1]
        img_norm = img.astype(np.float32) / 255.0 if img.max() > 1.0 else img.astype(np.float32)
        desc = hog(
            img_norm,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            channel_axis=None  # imagem em escala de cinza (sem canal de cor)
        )
        descritores.append(desc)
    return np.array(descritores)


def extrair_lbp(imagens, P = 8, R = 1.0):
    """
    Extrai descritores LBP de um conjunto de imagens

    Para cada imagem, o LBP é calculado pixel a pixel e então um histograma
    normalizado dos padrões é construído como vetor de características
    """
    descritores = []
    n_bins = P + 2
    for img in imagens:
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        lbp = local_binary_pattern(img_uint8, P=P, R=R, method='uniform')

        # Constrói histograma e normaliza para somar 1
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        descritores.append(hist)

    return np.array(descritores)


def extrair_hog_lbp(imagens):
    """
    Concatena os descritores HOG e LBP em um único vetor de características

    A concatenação permite que a CNN (ou qualquer classificador) explore tanto
    informações de borda/forma (HOG) quanto de textura (LBP) simultaneamente
    """
    hog_feats = extrair_hog(imagens)
    lbp_feats = extrair_lbp(imagens)
    return np.concatenate([hog_feats, lbp_feats], axis=1)