"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

Módulo principal do código: o ajuste dos parâmetros para realizar
o experimento deve ser feito diretamente na criação do objeto do MLP.
Além de inicializar o modelo, treiná-lo e exibir a matriz de confusão,
esse módulo também implementa algumas funções extras, como a criação de
um dicionário de mapeamento de letras para seus vetores (representações)
e uma função de holdout estratificado
"""

from mlp import *
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def criar_dict(y_col):
    """
    A função visa criar dicionário que mapeia cada letra para seu vetor
    Ex: {'A': [1,0,0, ...], 'B': [0,1,0, ...]}

    Recebe como parâmetros:
    1) y_col: coluna com as letras de cada amostra

    Retorna: Uma tupla (ordem_alfabetica, dict_conversao), onde ordem_alfabetica
    é a lista de letras únicas ordenada e dict_conversao mapeia cada letra ao seu
    vetor de tamanho 26
    """
    #Cria uma lista com as letras do alfabeto em ordem alfabética
    ordem_alfabetica = list(y_col.unique())
    ordem_alfabetica.sort()

    dict_conversao = {}
    for i in list(ordem_alfabetica):
        lista = [0] * 26
        indice = list(ordem_alfabetica).index(i)
        lista[indice] = 1
        dict_conversao[i] = lista

    return ordem_alfabetica, dict_conversao

def holdout_estratificado(x, valor_esperado_df,
                          rotulos, colunas_letras,
                          test_size=0.3, val_size=0.2,
                          seed=42):
    """
    Divide os dados em treino, validação e teste de forma estratificada por letra

    Recebe como parâmetros:
    1) x: dataframe com os atributos de todas as amostras
    2) valor_esperado_df: dataframe com a letra esperada de cada amostra
    3) rotulos: array com os vetores-alvo de cada amostra
    4) colunas_letras: Series com a letra de cada amostra
    5) test_size: proporção do total destinada ao teste (padrão: 0.3)
    6) val_size: proporção do treino destinada à validação (padrão: 0.2)
    7) seed: semente aleatória para reprodutibilidade (padrão: 42)

    Retorna: Tupla (treino_x, treino_y, rotulos_treino, val_x, val_y, rotulos_val,
    teste_x, teste_y, rotulos_teste) com os três conjuntos já separados
    """

    # Seed para garantir a reprodutibilidade do experimento
    np.random.seed(seed)

    total = len(x)
    n_teste_total = int(
        total * test_size)
    indices_fixos_teste = list(range(
        total - 130, total))
    indices_restantes = list(
        range(total - 130))

    # Quantas amostras ainda precisam ir para o teste além dos 130 fixos
    n_teste_extra = n_teste_total - 130

    indices_treino_total = []  # treino + validação, antes de separar a validação
    indices_teste_extra = []

    colunas_letras_restantes = colunas_letras.iloc[indices_restantes]

    for letra in sorted(colunas_letras_restantes.unique()):

        indices_letra = np.where(
            colunas_letras_restantes == letra)[0]
        np.random.shuffle(indices_letra)

        # Proporção da amostragem extra estratificada por classe
        n_extra_letra = max(1, int(len(indices_letra) * (
                n_teste_extra / len(indices_restantes))))

        indices_teste_extra.extend(
            indices_restantes[i] for i in indices_letra[:n_extra_letra])
        indices_treino_total.extend(
            indices_restantes[i] for i in indices_letra[n_extra_letra:])

    indices_teste = indices_fixos_teste + indices_teste_extra

    #  Separa a validação de dentro do treino, estratificada por letra
    indices_treino = []
    indices_val = []
    # Mapeia os índices de treino_total para suas respectivas letras
    letras_treino_total = colunas_letras.iloc[indices_treino_total].values
    indices_treino_total = np.array(indices_treino_total)

    for letra in sorted(np.unique(letras_treino_total)):
        idx_letra = indices_treino_total[letras_treino_total == letra]
        idx_letra = idx_letra.copy()
        np.random.shuffle(idx_letra)

        # Pelo menos 1 amostra de validação por letra (se houver mais de 1)
        n_val_letra = max(1, int(len(idx_letra) * val_size)) if len(idx_letra) > 1 else 0

        indices_val.extend(idx_letra[:n_val_letra].tolist())
        indices_treino.extend(idx_letra[n_val_letra:].tolist())

    treino_x = x.iloc[indices_treino, :]
    treino_y = valor_esperado_df.iloc[indices_treino, :]
    rotulos_treino = rotulos[indices_treino]

    val_x = x.iloc[indices_val, :]
    val_y = valor_esperado_df.iloc[indices_val, :]
    rotulos_val = rotulos[indices_val]

    teste_x = x.iloc[indices_teste, :]
    teste_y = valor_esperado_df.iloc[indices_teste, :]
    rotulos_teste = rotulos[indices_teste]

    print(f"\nDIVISÃO HOLDOUT ESTRATIFICADO (test_size={test_size}, val_size={val_size}, seed={seed})")
    print(f"Total: {total} amostras")
    print(f"Treino: {len(treino_x)} amostras ({round(len(treino_x)/total*100, 1)}%)")
    print(f"Validação: {len(val_x)} amostras ({round(len(val_x)/total*100, 1)}%)")
    print(f"Teste: {len(teste_x)} amostras ({round(len(teste_x)/total*100, 1)}%)")
    print(f"Teste = 130 fixos (finais) + {len(indices_teste_extra)} via estratificação")
    print(f"Validação = {val_size:.0%} do treino, estratificada por letra")
    print('\n')

    return (treino_x, treino_y,
            rotulos_treino, val_x,
            val_y, rotulos_val,
            teste_x, teste_y,
            rotulos_teste)

def main():
    """
    Ponto de entrada do programa: lê os dados, monta os rótulos, faz o holdout
    estratificado, instancia e treina a MLP, executa o teste
    e exibe a matriz de confusão

    Recebe como parâmetros: nenhum

    Retorna: None
    """

    BASE_DIR = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
    ENTRADAS = os.path.join(
        BASE_DIR, 'Entradas', 'CARACTERES COMPLETO')

    x = ler_arquivo_csv(os.path.join(ENTRADAS, 'X.txt'))
    y = ler_arquivo_csv(os.path.join(ENTRADAS, 'Y_letra.txt'))

    mlp = MLP(
        120,
        90,
        26,
        epocas=500,
        taxa_de_aprendizado=0.5,
        paciencia=20 # para se o erro de validação não melhorar por X épocas
    )

    colunas_letras = y[0]
    valor_esperado_df = y[[0]]

    letras, dict_conversao = criar_dict(colunas_letras)

    rotulos = np.array([dict_conversao[letra] for letra in colunas_letras])

    # Quando o arquivo é lido a coluna 120 está vazia
    x = x.drop(columns={120})

    (treino_x, treino_y, rotulos_treino,
     val_x, val_y, rotulos_val,
     teste_x, teste_y, rotulos_teste) = holdout_estratificado(
        x, valor_esperado_df,
        rotulos, colunas_letras,
        test_size=0.3, val_size=0.2,
        seed=42
    )

    # Treina passando o conjunto de validação ativa a parada antecipada (Sem esse conjunto a parada
    # antecipada não é ativada)
    mlp.fit(treino_x, rotulos_treino,
            val_dados=val_x, val_rotulos=rotulos_val)

    # Recebe os resultados do teste da MLP
    resultados = mlp.teste(teste_x, rotulos_teste,
                           letras, teste_y)

    # Gera e exibe a matriz de confusão
    mlp.matriz_confusao(resultados, letras)


if __name__ == '__main__':
    main()