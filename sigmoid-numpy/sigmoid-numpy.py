import numpy as np

def sigmoid(x):
    x_array = np.asanyarray(x, dtype= float)

    return 1 / (1 + np.exp(-x_array))