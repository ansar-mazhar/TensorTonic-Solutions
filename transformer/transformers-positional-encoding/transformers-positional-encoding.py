import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    seq_length: Number of tokens in the sequence (L)
    d_model: Dimension of the embedding vectors (D)
    """
    # 1. Initialize a matrix of zeros with shape (seq_length, d_model)
    pe = np.zeros((seq_length, d_model))
    
    # 2. Create a column vector of positions [0, 1, 2, ..., seq_length-1]
    position = np.arange(0, seq_length).reshape(-1, 1)
    
    # 3. Calculate the division term (the 'step' in the frequencies)
    # We use exp and log for numerical stability: 10000^(2i/d_model)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # 4. Apply sine to even indices (0, 2, 4...)
    pe[:, 0::2] = np.sin(position * div_term)
    
    # 5. Apply cosine to odd indices (1, 3, 5...)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe