import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    """The forward method converts input token IDs into their corresponding embedding vectors using nn.Embedding and then scales those embeddings by √d_model, 
       producing the properly scaled representations required by the Transformer. """
    def forward(self,x):
        return self.embedding(x)* math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model :int, seq_len :int, dropout :float)-> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len,d_model)
        # create a vector of shape(seq_len)
        

    


    

