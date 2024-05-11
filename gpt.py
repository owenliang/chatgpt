from torch import Module
from torch import nn

class GPT(Module):
    def __init__(self,d_model,nhead,feedforward,vocab_size):
        super().__init__()
        
        # positional encoding...
        
        # vocab embedding
        self.emb=nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        
        # decoder-only transformer (self-attention)
        self.dec_layers=nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=feedforward,batch_first=True) for _ in range(5)
        ])
        # next token probability
        self.prob_linear=nn.Linear(d_model,vocab_size)
    
    def forward(self,batch_x,attention_mask):
        pass