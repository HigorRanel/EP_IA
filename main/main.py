import numpy as np
from utils import ler_arquivo_csv
from mlp import *
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def criar_dict(y_col):
    """
    A função visa criar dicionário que mapeia cada letra para seu vetor
    Ex: {'A': [1,0,0, ...], 'B': [0,1,0, ...]}
    """
    ordem_alfabetica = list(y_col.unique())
    ordem_alfabetica.sort()

    dict_conversao = {}
    for i in list(ordem_alfabetica):
        lista = [0] * 26
        indice = list(ordem_alfabetica).index(i)
        lista[indice] = 1
        dict_conversao[i] = lista

    return ordem_alfabetica, dict_conversao

def div_teste_treino (x, valor_esperado_df, rotulos, colunas_letras, test_size=0.3, seed=42):
    # A semente é necessária para garantir a reprodutibilidade do experimento aleatório de seleção
    # dos dados que vão para teste ou para treino
    np.random.seed(seed)

    # Listas dos índices dos dataframes que serão utilizados para teste ou para treino
    indices_treino = []
    indices_teste = []

    # Para cada letra, embaralhamos os índices dos dados que representam essa letra e dividimos de
    # acordo com a proporção dada como parâmetro (test_size).
    for letra in sorted(colunas_letras.unique()):
        # Pega todos os índices desta letra no dataset
        indices_letra = np.where(colunas_letras == letra)[0]

        # Embaralha para não pegar sempre as mesmas versões
        np.random.shuffle(indices_letra)

        # Calcula o ponto da lista indices_letra em que se fará a divisão entre teste e treino
        n_teste = max(1, int(len(indices_letra) * test_size))

        indices_teste.extend(indices_letra[:n_teste])
        indices_treino.extend(indices_letra[n_teste:])

    treino_x = x.iloc[indices_treino, :]
    treino_y = valor_esperado_df.iloc[indices_treino, :]
    rotulos_treino = rotulos[indices_treino]

    teste_x = x.iloc[indices_teste, :]
    teste_y = valor_esperado_df.iloc[indices_teste, :]
    rotulos_teste = rotulos[indices_teste]

    return treino_x, treino_y, rotulos_treino, teste_x, teste_y, rotulos_teste

def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ENTRADAS = os.path.join(BASE_DIR, 'Entradas', 'CARACTERES COMPLETO')

    x = ler_arquivo_csv(os.path.join(ENTRADAS, 'X.txt'))
    y = ler_arquivo_csv(os.path.join(ENTRADAS, 'Y_letra.txt'))
    mlp=MLP(120, 100, 26, epocas=500, taxa_de_aprendizado=0.8, limiar_erro=0.01)

    colunas_letras = y[0]
    valor_esperado_df = y[[0]]

    letras, dict_conversao = criar_dict(colunas_letras)
    
    rotulos = np.array([dict_conversao[letra] for letra in colunas_letras])

    #X_np = x.values
    # print(x.iloc[:, 120])
    x=x.drop(columns={120})
    # print(x.iloc[:, 119])

    treino_percent=int(0.6*x.shape[0])
    teste_percent=int(0.2*x.shape[0])

    # treino_x=x.iloc[0:treino_percent,:]
    # treino_y=valor_esperado_df.iloc[0:treino_percent,:]
    # rotulos_treino=rotulos[0:treino_percent]
    #
    # teste_x=x.iloc[treino_percent:treino_percent+teste_percent, :]
    # teste_y=valor_esperado_df.iloc[treino_percent:treino_percent+teste_percent, :]
    # rotulos_teste=rotulos[treino_percent:treino_percent+teste_percent]

    # === ALTERAÇÃO PARA DADOS REDUZIDOS ===
    # qtd_treino = 5  # <--- Mude aqui a quantidade de treino
    # qtd_teste = 2  # <--- Mude aqui a quantidade de teste
    #
    # treino_x = x.iloc[0:qtd_treino, :]
    # treino_y = valor_esperado_df.iloc[0:qtd_treino, :]
    # rotulos_treino = rotulos[0:qtd_treino]
    #
    # teste_x = x.iloc[qtd_treino: qtd_treino + qtd_teste, :]
    # teste_y = valor_esperado_df.iloc[qtd_treino: qtd_treino + qtd_teste, :]
    # rotulos_teste = rotulos[qtd_treino: qtd_treino + qtd_teste]
    # # =======================================

    treino_x, treino_y, rotulos_treino, teste_x, teste_y, rotulos_teste = div_teste_treino(x, valor_esperado_df,
                                                                                         rotulos,
                                                                                           colunas_letras,
                                                                                           test_size=0.3,
                                                                                           seed=42)
    print(f"Treino: {len(treino_x)} amostras | Teste: {len(teste_x)} amostras")

    mlp.fit(treino_x, rotulos_treino, limiar_erro=0.01)
    resultados = mlp.teste(teste_x, rotulos_teste, letras, teste_y)

    # Gera e exibe a matriz de confusão
    mlp.matriz_confusao(resultados, letras)

    # print(treino_x)
    # print(treino_y)
    # print(teste_x)
    # print(teste_y)

if __name__ == '__main__':
    main()