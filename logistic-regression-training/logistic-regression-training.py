import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    for _ in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        error = p - y
        dw = (1 / N) * np.dot(X.T, error)
        db = (1 / N) * np.sum(error)
        w -= lr * dw
        b -= lr * db
    return w, b 
            
        
    
    
  