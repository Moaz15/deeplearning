import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Takes token IDs (integers) and converts them into vectors of size d_model.

    In a Transformer, the model cannot work directly with token IDs like [5, 20, 5];
    it needs continuous vectors. This module provides those vectors via nn.Embedding,
    then scales them by sqrt(d_model) (as done in the original Transformer paper).
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        # Initialize nn.Module internals (parameter/submodule registration, etc.)
        super().__init__()

        # d_model = size of each embedding vector (Transformer hidden dimension)
        self.d_model = d_model

        # vocab_size = number of unique tokens in the vocabulary (max token ID + 1)
        self.vocab_size = vocab_size

        # Learnable embedding table (weight matrix) of shape: (vocab_size, d_model)
        # Each token id t maps to the row: embedding.weight[t]  -> shape: (d_model,)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        x: token IDs (integers)
           shape can be:
             - (seq_len,)                  for one sequence
             - (batch_size, seq_len)       for a batch of sequences

        Step 1: embedding lookup
            self.embedding(x) replaces each token ID with its vector of length d_model.
            Output shape becomes:
              - (seq_len, d_model)                 if input was (seq_len,)
              - (batch_size, seq_len, d_model)     if input was (batch_size, seq_len)

        Step 2: scale embeddings
            Multiply by sqrt(d_model) to match the scaling used in the Transformer paper.
            This helps keep the magnitude of the representations well-behaved as d_model changes
            (useful for stable training, especially before attention/positional encoding).
        """

        # Convert token IDs -> embedding vectors, then scale
        return self.embedding(x) * math.sqrt(self.d_model)



    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/ d_model))

        pe[:, 0::2] = torch.sin(position* div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)










    


    

