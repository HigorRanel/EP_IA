"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

Módulo de extração de descritores de imagem.

Descritores implementados:
- HOG (Histogram of Oriented Gradients): captura a distribuição de gradientes de
  intensidade/bordas em regiões locais da imagem. É robusto a pequenas variações
  de iluminação e deformações geométricas.

- LBP (Local Binary Patterns): codifica a textura local de cada pixel comparando-o
  com seus vizinhos. É computacionalmente leve e invariante a transformações
  monotônicas de escala de cinza.

Biblioteca utilizada: scikit-image
Citação recomendada pelos autores:
  Stéfan van der Walt et al., "scikit-image: image processing in Python",
  PeerJ 2:e453, 2014. http://dx.doi.org/10.7717/peerj.453
"""

import numpy as np
from skimage.feature import hog, local_binary_pattern


def extrair_hog(imagens: np.ndarray) -> np.ndarray:
    """
    Extrai descritores HOG de um conjunto de imagens.

    Parâmetros do HOG escolhidos:
    - orientations=8: número de bins do histograma de gradientes por célula.
      Valor padrão amplamente usado; suficiente para capturar formas de roupas.
    - pixels_per_cell=(4, 4): tamanho de cada célula local. Células menores
      capturam detalhes mais finos; 4x4 é adequado para imagens 28x28.
    - cells_per_block=(2, 2): normalização ocorre em blocos de 2x2 células,
      tornando o descritor mais robusto a variações de contraste.

    Args:
        imagens: array de shape (N, 28, 28) com valores em [0, 255] ou [0, 1].

    Returns:
        array de shape (N, D) onde D é o comprimento do vetor HOG por imagem.
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


def extrair_lbp(imagens: np.ndarray, P: int = 8, R: float = 1.0) -> np.ndarray:
    """
    Extrai descritores LBP de um conjunto de imagens.

    Para cada imagem, o LBP é calculado pixel a pixel e então um histograma
    normalizado dos padrões é construído como vetor de características.

    Parâmetros do LBP escolhidos:
    - P=8: número de pontos vizinhos amostrados em volta do pixel central.
    - R=1.0: raio do círculo de vizinhança (1 pixel = vizinhança imediata).
    - method='uniform': considera apenas padrões "uniformes" (com no máximo 2
      transições binárias), reduzindo o vetor de 2^P para P+2 dimensões e
      aumentando a robustez a ruído.

    Args:
        imagens: array de shape (N, 28, 28).
        P: número de pontos vizinhos.
        R: raio da vizinhança.

    Returns:
        array de shape (N, P+2) com histogramas LBP normalizados.
    """
    descritores = []
    n_bins = P + 2  # padrões uniformes: 0..P + 1 bin para não-uniformes

    for img in imagens:
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        lbp = local_binary_pattern(img_uint8, P=P, R=R, method='uniform')

        # Constrói histograma e normaliza para somar 1
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        descritores.append(hist)

    return np.array(descritores)


def extrair_hog_lbp(imagens: np.ndarray) -> np.ndarray:
    """
    Concatena os descritores HOG e LBP em um único vetor de características.

    A concatenação permite que a CNN (ou qualquer classificador) explore tanto
    informações de borda/forma (HOG) quanto de textura (LBP) simultaneamente.

    Args:
        imagens: array de shape (N, 28, 28).

    Returns:
        array de shape (N, D_hog + D_lbp).
    """
    hog_feats = extrair_hog(imagens)
    lbp_feats = extrair_lbp(imagens)
    return np.concatenate([hog_feats, lbp_feats], axis=1)
