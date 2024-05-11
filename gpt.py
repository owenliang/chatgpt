from torch import nn
import torch 
from emb import EmbeddingWithPosition

class GPT(nn.Module):
    def __init__(self,d_model,nhead,feedforward,vocab_size):
        super().__init__()
        
        # positional encoding...
        self.emb=EmbeddingWithPosition(vocab_size=vocab_size,dim=d_model)
        
        # decoder-only transformer (self-attention)
        self.dec_blocks=nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=feedforward,batch_first=True) for _ in range(5)
        ])
        # next token probability
        self.prob_linear=nn.Linear(d_model,vocab_size)
    
    def forward(self,x,padding_mask): # x:(batch,seq)
        # 注意力遮挡
        src_mask=torch.triu(torch.ones(x.size()[1],x.size()[1]),diagonal=1)
        # embedding
        x=self.emb(x)
        # decoder
        for block in self.dec_blocks:
            x=block(x,src_mask=src_mask,src_key_padding_mask=padding_mask)
        # logits
        logits=self.prob_linear(x)
        return logits

if __name__=='__main__':
    # 分词器
    from bpe import BPETokenizer
    tokenizer=BPETokenizer()
    tokenizer.load('tokenizer.bin')
    
    # 模拟输入
    x=torch.randint(0,tokenizer.vocab_size(),(5,30))
    padding=torch.zeros(5,30)
    
    # GPT模型
    gpt=GPT(d_model=64,nhead=2,feedforward=128,vocab_size=tokenizer.vocab_size())
    y=gpt(x,padding)
    print(y.shape)