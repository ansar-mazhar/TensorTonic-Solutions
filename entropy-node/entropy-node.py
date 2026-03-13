import numpy as np


def entropy_node(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    unique_values, counts = np.unique(labels, return_counts = True)
    probs = counts / n
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)
