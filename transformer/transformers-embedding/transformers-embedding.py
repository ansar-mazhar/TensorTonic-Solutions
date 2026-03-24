import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    vocab_size: Total number of unique words in your dictionary.
    d_model: The size of each embedding vector (e.g., 512).
    """
    # nn.Embedding is essentially a lookup table of size (vocab_size, d_model)
    return nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # 1. Pass the token IDs through the embedding layer to get vectors
    x = embedding(tokens)
    
    # 2. Scale the embeddings by sqrt(d_model)
    # This is a standard practice in Transformers (like Attention Is All You Need)
    # to prevent gradients from vanishing/exploding during training.
    return x * math.sqrt(d_model)