"""
Nomes e Nº USP:
1. Bruno Hideo Ioneda - NUSP: 15573619
2. Guilherme Samuel Lemos Segura - NUSP: 15575611
3. Higor Ranel Viani Lopes - NUSP: 15552946
4. João de Melo Fantini - NUSP: 15462550
5. Luiz Vicente Neto - NUSP: 14593054

Busca de parâmetros (grid search) para a MLP.
"""

import os
import sys
import csv
import time
import io
import contextlib
from datetime import datetime

import numpy as np
from main import criar_dict, holdout_estratificado, NEURONIOS_SAIDA
from utils import ler_arquivo_csv


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp import MLP

# Configuração de parâmetros

NEURONIOS = [30, 60, 90, 120] # neurônios na camada escondida
TAXAS = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]  # taxa de aprendizado
INICIALIZACOES_PESOS = ['aleatorio', 'Xavier']  # regimes de inicialização dos pesos
INICIALIZACOES_BIAS = {'aleatorio': 'aleatorio', 'Xavier': 'zero'}
EPOCAS = 250  # máximo (parada antecipada corta antes)
PACIENCIA = 20 # paciência da parada antecipada
_SEED = 42  # semente única para reprodutibilidade de toda a busca


@contextlib.contextmanager
def _silenciar_stdout():
    """
    Suprime temporariamente o stdout (prints da MLP) durante a execução de
    cada combinação, para que o console mostre apenas o progresso da busca
    """
    stdout_original = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield
    finally:
        sys.stdout = stdout_original


def _acuracia(mlp, dados, rotulos):
    """
    Calcula a acurácia da MLP sobre um conjunto (apenas feedforward), comparando o
    argmax da saída com o argmax do rótulo one-hot. Não grava arquivos nem altera
    pesos, então pode ser usada para medir validação e teste durante a busca.

    Recebe como parâmetros:
    1) mlp: a rede já treinada
    2) dados: dataframe com as amostras a avaliar
    3) rotulos: array com os vetores-alvo (one-hot) correspondentes

    Retorna: a fração de amostras classificadas corretamente (0.0 se vazio)
    """
    n = dados.shape[0]
    if n == 0:
        return 0.0
    acertos = 0
    for i in range(n):
        mlp.forward(dados.iloc[i])
        if np.argmax(mlp.y) == np.argmax(rotulos[i]):
            acertos += 1
    return acertos / n


def buscar_parametros(treino_x, rotulos_treino,
                      val_x, rotulos_val,
                      teste_x, rotulos_teste,
                      letras, valor_esperado_teste,
                      n_entradas, n_saidas,
                      neuronios=NEURONIOS, taxas=TAXAS,
                      inicializacoes=INICIALIZACOES_PESOS,
                      epocas=EPOCAS, paciencia=PACIENCIA,
                      caminho_csv=None):
    """
    Executa a busca em grade e salva o CSV-resumo

    A seleção da melhor combinação é feita pela acurácia de VALIDAÇÃO, nunca pela de
    teste: o conjunto de teste é só reportado ao final, para que continue sendo uma
    estimativa imparcial de generalização (evita vazamento de dados).

    Reprodutibilidade: o RNG global é re-semeado (np.random.seed(_SEED)) antes de
    cada combinação, de modo que todas partam do mesmo init de pesos/bias e do mesmo
    embaralhamento por época. Assim a comparação entre hiperparâmetros é justa e o
    resultado é determinístico, independente da ordem da grade.

    Recebe como parâmetros:
    1) treino_x: dataframe com as amostras de treino
    2) rotulos_treino: array com os vetores-alvo de treino
    3) val_x: dataframe com as amostras de validação
    4) rotulos_val: array com os vetores-alvo de validação
    5) teste_x: dataframe com as amostras de teste
    6) rotulos_teste: array com os vetores-alvo de teste
    7) letras: lista de letras do alfabeto
    8) valor_esperado_teste: dataframe com a letra esperada de cada amostra de teste
    9) n_entradas: nº de neurônios da camada de entrada
    10) n_saidas: nº de neurônios da camada de saída
    11) neuronios: lista de tamanhos de camada oculta a testar
    12) taxas: lista de taxas de aprendizado a testar
    13) inicializacoes: lista de regimes de inicialização de pesos a testar
    14) epocas: nº máximo de épocas por combinação
    15) paciencia: paciência da parada antecipada
    16) caminho_csv: caminho do CSV de saída

    Retorna: Lista de dicionários, um por combinação testada, com os hiperparâmetros,
    as acurácias de validação e teste, o nº de épocas treinadas, o tempo e os erros finais
    """
    if caminho_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_csv = f"busca_parametros_{ts}.csv"

    combinacoes = [(n, t, ini)
                   for n in neuronios
                   for t in taxas
                   for ini in inicializacoes]
    total = len(combinacoes)
    resultados = []

    print(f"\nBUSCA DE PARÂMETROS: {total} combinações "
          f"({len(neuronios)} neurônios x {len(taxas)} taxas x {len(inicializacoes)} inits)")
    print(f"Épocas máx.: {epocas} | Paciência: {paciencia}\n")

    for idx, (n_oculta, taxa, ini_pesos) in enumerate(combinacoes, start=1):
        print(f"[{idx:>2}/{total}] neurônios={n_oculta:>3}, taxa={taxa}, "
              f"init={ini_pesos:>9} " f"init_bias={INICIALIZACOES_BIAS[ini_pesos]}... ", end="", flush=True)

        inicio = time.time()
        # Re-semeia o RNG global antes de CADA combinação. Como o stream do NumPy é
        # compartilhado e sequencial, sem isso o init de pesos/bias de cada MLP
        # dependeria de quantos números aleatórios as combinações anteriores
        # consumiram (inclusive de quantas épocas a parada antecipada treinou).
        # Re-semeando, toda combinação parte do MESMO init e do mesmo embaralhamento
        # por época, tornando a comparação de hiperparâmetros justa e reprodutível.
        np.random.seed(_SEED)
        # Toda a saída da MLP (barra, resumo, matriz) é ocultado aqui
        with _silenciar_stdout():
            mlp = MLP(
                n_entradas, n_oculta, n_saidas,
                epocas=epocas,
                taxa_de_aprendizado=taxa,
                paciencia=paciencia,
                ini_pesos=ini_pesos,
                ini_bias=INICIALIZACOES_BIAS[ini_pesos]
            )

            # Treina o MLP
            mlp.fit(treino_x, rotulos_treino, val_dados=val_x, val_rotulos=rotulos_val)

            # Acurácia de VALIDAÇÃO: critério de SELEÇÃO dos hiperparâmetros
            acuracia_val = _acuracia(mlp, val_x, rotulos_val)

            # Testa o MLP. A acurácia de teste é apenas REPORTADA, nunca usada para
            # escolher a combinação (evita vazamento do conjunto de teste)
            res_teste = mlp.teste(teste_x, rotulos_teste, letras, valor_esperado_teste)

        # Tempo de execução (tempo de treino + tempo de teste) do MLP:
        tempo = time.time() - inicio

        # Acurácia a partir dos resultados de teste
        acertos = sum(1 for r in res_teste
                      if str(r['esperado']).casefold() == str(r['previsto']).casefold())
        total_teste = len(res_teste)
        acuracia = acertos / total_teste if total_teste else 0.0

        # Nº de épocas efetivamente treinadas e erros finais
        epocas_treinadas = len(mlp.erros)
        eqm_treino_final = mlp.erros[-1] if mlp.erros else None
        eqm_val_final = mlp.erros_val[-1] if mlp.erros_val else None

        linha = {
            'neuronios_ocultos': n_oculta,
            'taxa_aprendizado': taxa,
            'ini_pesos': ini_pesos,
            'ini_bias': INICIALIZACOES_BIAS[ini_pesos],
            'epocas_treinadas': epocas_treinadas,
            'epocas_max': epocas,
            'acuracia_val': round(acuracia_val, 4),
            'acuracia_teste': round(acuracia, 4),
            'acertos': acertos,
            'total_teste': total_teste,
            'tempo_segundos': round(tempo, 2),
            'eqm_treino_final': round(eqm_treino_final, 6) if eqm_treino_final is not None else '',
            'eqm_val_final': round(eqm_val_final, 6) if eqm_val_final is not None else '',
        }
        resultados.append(linha)

        print(f"val={acuracia_val*100:5.2f}%  teste={acuracia*100:5.2f}%  "
              f"épocas={epocas_treinadas:>3}  tempo={tempo:6.1f}s")

    # Salva o CSV de resumo
    colunas = ['neuronios_ocultos', 'taxa_aprendizado', 'ini_pesos',
               'ini_bias',
               'epocas_treinadas', 'epocas_max',
               'acuracia_val', 'acuracia_teste', 'acertos',
               'total_teste', 'tempo_segundos',
               'eqm_treino_final', 'eqm_val_final']
    with open(caminho_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        for linha in resultados:
            writer.writerow(linha)

    # Identifica a melhor combinação pela acurácia de VALIDAÇÃO (desempate: menor
    # tempo). O teste é só reportado, para não enviesar a seleção (sem vazamento)
    melhor_acuracia_val = max(r['acuracia_val'] for r in resultados)
    candidatos = [r for r in resultados if r['acuracia_val'] == melhor_acuracia_val]
    melhor = min(candidatos, key=lambda r: r['tempo_segundos'])
    print(f"\nBUSCA CONCLUÍDA")
    print(f"CSV salvo em: {caminho_csv}")
    print(f"Melhor combinação (por validação): neurônios={melhor['neuronios_ocultos']}, "
          f"taxa={melhor['taxa_aprendizado']}, init={melhor['ini_pesos']} | "
          f"acurácia val={melhor['acuracia_val']*100:.2f}% "
          f"teste={melhor['acuracia_teste']*100:.2f}% "
          f"em {melhor['tempo_segundos']}s\n")

    return resultados

if __name__ == '__main__':
    # Garante a reprodutibilidade de toda a busca
    # e deve ser definida antes de qualquer outra inicialização
    np.random.seed(_SEED)

    BASE_DIR = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
    ENTRADAS = os.path.join(
        BASE_DIR, 'Entradas', 'CARACTERES COMPLETO')

    x = ler_arquivo_csv(os.path.join(ENTRADAS, 'X.txt'))
    y = ler_arquivo_csv(os.path.join(ENTRADAS, 'Y_letra.txt'))
    colunas_letras = y[0]; valor_esperado_df = y[[0]]
    letras, dconv = criar_dict(colunas_letras)
    rotulos = np.array([dconv[l] for l in colunas_letras])
    x = x.drop(columns={120})
    (trx, try_, rtr, vx, vy, rval, tex, tey, rte) = holdout_estratificado(
        x, valor_esperado_df, rotulos, colunas_letras, test_size=0.3, val_size=0.2)

    buscar_parametros(trx, rtr, vx, rval, tex, rte, letras, tey,
                      n_entradas=120, n_saidas=26)