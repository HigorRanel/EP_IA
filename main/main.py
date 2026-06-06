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

_NEURONIOS_SAIDA = 26
_SEED = 42

def reduzir_treino(
        treino_x,
        rotulos_treino,
        colunas_letras_treino,
        max_por_letra=5,
):
    """
    Mantém no máximo max_por_letra amostras de cada letra no treinamento
    Buscamos com essa função entender como o modelo generaliza

    Parâmetros:
    1) treino_x: dataframe de treino
    2) rotulos_treino: array one-hot correspondente
    3) colunas_letras_treino: Series/array com a letra de cada amostra de treino
    4) max_por_letra: maximo de amostras por letra

    Retorna: Uma tupla com 2 conjuntos reajustados:
    treino_x_reduzido: Os dados de treino com a quantidade de exemplos limitada ao valor definido
    rotulos_treino_reduzido: As respostas que acompanham esse grupo
    """

    letras = np.array(colunas_letras_treino)
    selecionados = []

    for letra in sorted(set(letras)):
        index = np.where(letras == letra)[0]
        np.random.shuffle(index)

        selecionados.extend(index[:max_por_letra])

    selecionados = sorted(selecionados)

    return treino_x.iloc[selecionados, :], rotulos_treino[selecionados]


def split_nao_estratificado(
        x,
        rotulos,
        colunas_letras,
):
    """
    Divide treino com proporção distintas para cada letra, sem estratificar.
    Inspirado no cenário citado pela professora.

    Parâmetros:
    1) x: dataframe completo
    2) rotulos: array one-hot completo
    3) colunas_letras: array com a letra de cada amostra

    Retorna: uma tupla com 4 elementos
    treino_x: Os dados separados para ensinar o modelo, 30 a 80%
    rotulos_treino: As respostas do conjunto de treino
    teste_x: O restante dos dados para avaliar o modelo
    rotulos_teste: As respostas do conjunto de teste
    """
    letras = np.array(colunas_letras)
    index_treino, index_teste = [], []

    for letra in sorted(set(letras)):
        index = np.where(
            letras == letra)[0]
        np.random.shuffle(index)

        # fração de treino sorteada por letra
        #  varia entre 30% e 80%
        frac_treino = np.random.uniform(0.3, 0.8)
        corte = max(1, int(
            len(index) * frac_treino))
        index_treino.extend(index[:corte])
        index_teste.extend(index[corte:])

    index_treino, index_teste = sorted(index_treino), sorted(index_teste)

    return (
        x.iloc[index_treino, :], rotulos[index_treino],
        x.iloc[index_teste, :], rotulos[index_teste]
    )

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
        lista = [0] * _NEURONIOS_SAIDA
        indice = list(ordem_alfabetica).index(i)
        lista[indice] = 1
        dict_conversao[i] = lista

    return ordem_alfabetica, dict_conversao

def holdout_estratificado(x, valor_esperado_df,
                          rotulos, colunas_letras,
                          test_size=0.3, val_size=0.2):
    """
    Divide os dados em treino, validação e teste de forma estratificada por letra

    Recebe como parâmetros:
    1) x: dataframe com os atributos de todas as amostras
    2) valor_esperado_df: dataframe com a letra esperada de cada amostra
    3) rotulos: array com os vetores-alvo de cada amostra
    4) colunas_letras: Series com a letra de cada amostra
    5) test_size: proporção do total destinada ao teste (padrão: 0.3)
    6) val_size: proporção do treino destinada à validação (padrão: 0.2)

    Retorna: Tupla (treino_x, treino_y, rotulos_treino, val_x, val_y, rotulos_val,
    teste_x, teste_y, rotulos_teste) com os três conjuntos já separados
    """

    # Seed para garantir a reprodutibilidade do experimento

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

    print(f"\nDIVISÃO HOLDOUT ESTRATIFICADO (test_size={test_size}, val_size={val_size}, seed={_SEED})")
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

def funde_classes(coluna_letras, fundir):
    if not fundir:
        return coluna_letras
    
    return coluna_letras.replace(fundir)


def main():
    """
    Ponto de entrada do programa: lê os dados, monta os rótulos, faz o holdout
    estratificado, instancia e treina a MLP, executa o teste
    e exibe a matriz de confusão

    Recebe como parâmetros: nenhum

    Retorna: None
    """

    # Seed para garantir reprodutibilidade
    np.random.seed(_SEED)

    BASE_DIR = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
    ENTRADAS = os.path.join(
        BASE_DIR, 'Entradas', 'CARACTERES COMPLETO')

    x = ler_arquivo_csv(os.path.join(ENTRADAS, 'X.txt'))
    y = ler_arquivo_csv(os.path.join(ENTRADAS, 'Y_letra.txt'))

    colunas_letras = y[0]
    # o dict seguinte serve para fundir duas classes em uma
    # Exemplo de input: {'K':'X'}
    fundir = {}

    colunas_letras = funde_classes(colunas_letras, fundir)

    valor_esperado_df = colunas_letras.to_frame()

    letras, dict_conversao = criar_dict(colunas_letras)

    rotulos = np.array([dict_conversao[letra] for letra in colunas_letras])

    # Quando o arquivo é lido a coluna 120 está vazia
    x = x.drop(columns={120})

    (treino_x, treino_y, rotulos_treino,
     val_x, val_y, rotulos_val,
     teste_x, teste_y, rotulos_teste) = holdout_estratificado(
        x, valor_esperado_df,
        rotulos, colunas_letras,
        test_size=0.3, val_size=0.2
    )

    # Re-semeia a seed para garantir reprodutibilidade e igualdade na comporação dos hiperparâmetros
    np.random.seed(_SEED)
    mlp = MLP(
        120,
        60,
        _NEURONIOS_SAIDA,
        epocas=150,
        taxa_de_aprendizado=0.9,
        paciencia=20,  # para se o erro de validação não melhorar por X épocas
        ini_pesos='Xavier',
        ini_bias='zero'
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

    # Variações Autorais:

    # # Variação: treino reduzido
    # # letra de cada amostra de treino
    # colunas_letras_treino = treino_y.iloc[:, 0]
    # treino_x_red, rot_red = reduzir_treino(treino_x, rotulos_treino,
    #                                        colunas_letras_treino, max_por_letra=5)
    # # Re-semeia após o sorteio da redução e logo antes de construir o modelo,
    # # para partir do mesmo init/embaralhamento do modelo principal (comparação justa)
    # np.random.seed(_SEED)
    # mlp_a = MLP(120, 90, _NEURONIOS_SAIDA, epocas=150, taxa_de_aprendizado=0.5,
    #             paciencia=10, ini_pesos='xavier', ini_bias='zero')
    # mlp_a.fit(treino_x_red, rot_red, val_dados=val_x, val_rotulos=rotulos_val)
    # res_red = mlp_a.teste(teste_x, rotulos_teste, letras, teste_y)
    # print("\nTREINO REDUZIDO (5 por letra)")
    # mlp_a.matriz_confusao(res_red, letras)

    ###############################################################################

    # # Variação: split não estratificado
    # trx, rtr, tex, rte = split_nao_estratificado(x, rotulos, colunas_letras)
    # tey = valor_esperado_df.iloc[tex.index, :]
    #
    # # separa validação de dentro do treino (sem estratificar) p/ ativar a parada antecipada
    # idx = np.arange(len(trx))
    # np.random.shuffle(idx)
    # corte = int(len(trx) * 0.2)
    # idx_val, idx_tr = idx[:corte], idx[corte:]
    #
    # trx2, rtr2 = trx.iloc[idx_tr, :], rtr[idx_tr]
    # vx, rval = trx.iloc[idx_val, :], rtr[idx_val]
    #
    # # Re-semeia após os sorteios do split e logo antes de construir o modelo,
    # # para partir do mesmo init/embaralhamento do modelo principal (comparação justa)
    # np.random.seed(_SEED)
    # mlp_b = MLP(120, 90, _NEURONIOS_SAIDA, epocas=150,
    #             taxa_de_aprendizado=0.5, paciencia=10,
    #             ini_pesos='xavier', ini_bias='zero')
    #
    # mlp_b.fit(trx2, rtr2, val_dados=vx, val_rotulos=rval)
    # res_nestrat = mlp_b.teste(tex, rte, letras, tey)
    # print("\nSPLIT NÃO ESTRATIFICADO")
    # mlp_b.matriz_confusao(res_nestrat, letras)


if __name__ == '__main__':
    main()