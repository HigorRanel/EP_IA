import pandas as pd
import numpy as np

X = pd.read_csv(r'C:\Users\luizv\PyCharmMiscProject\EP_IA\Entradas\CARACTERES COMPLETO\X.txt', header=None)
Y = pd.read_csv(r'C:\Users\luizv\PyCharmMiscProject\EP_IA\Entradas\CARACTERES COMPLETO\Y_letra.txt', header=None)
print(X[120].value_counts())
X[120]=Y[0]
print(X[120].value_counts())
