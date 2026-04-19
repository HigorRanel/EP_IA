import numpy as np

def step_t(inpt:float, num:float)->float:
    if inpt >= num:
        return 1
    else:
        return 0

def lin_part(inpt:float)->float:
    if inpt<=-1/2:
        return -1
    elif inpt<1/2:
        return inpt
    else:
        return 1

def sigmoid(inpt:float)->float:
    return 1/(1+np.exp(-inpt))

def derivada_sigmoid(inpt:float)->float:
    # A derivida da sigmoid é matematicamente definida como:
    return sigmoid(inpt) * (1 - sigmoid(inpt))

def relu(x):
    return np.maximum(0, x)