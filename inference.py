from gpt import GPT
from config import *
import torch 
from tokenizer import BPETokenizer
import torch.nn.functional as F
import random

# 设备
DEVICE='cuda' if torch.cuda.is_available() else 'cpu' 

# 分词器
tokenizer=BPETokenizer()  
tokenizer.load('tokenizer.bin')

# 加载模型
model=GPT(d_model=GPT_DIM,nhead=GPT_HEAD,feedforward=GPT_FF,vocab_size=tokenizer.vocab_size(),seq_max_len=MAX_SEQ_LEN).to(DEVICE) # 模型
try:  
    checkpoint=torch.load('checkpoint.bin')
    model.load_state_dict(checkpoint['model'])
except:
    pass

model.eval()

# 可能的结束符
eos_ids,_=tokenizer.encode(EOS)
pad_ids,_=tokenizer.encode(PAD)
im_end_ids,_=tokenizer.encode(IM_END)

def chat(query):
    global tokenizer,model

    inputs=f'{BOS}{IM_START}user\n{query}\n{IM_END}\n{IM_START}assistant\n' if GPT_MODE=='chat' else f'{BOS}{query}'
    ids,_=tokenizer.encode(inputs)
    
    while len(ids)<MAX_SEQ_LEN:
        batch_ids=torch.tensor([ids],dtype=torch.long).to(DEVICE)
        batch_paddding_mask=torch.tensor([[0]*len(ids)],dtype=torch.bool).to(DEVICE)
        
        with torch.no_grad():
            logits=model(batch_ids,batch_paddding_mask) # (batch,seq,vocab)
            # 多样性控制
            logits=logits[0,-1,:]/TEMPERATURE
            topk_logits,topk_ids=torch.topk(logits,k=TOP_K)
            topk_logits,topk_ids=topk_logits.cpu(),topk_ids.cpu()
            # 从topk中随机挑1个token
            topk_probs=F.softmax(topk_logits,dim=-1)
            rnd=random.random()
            cumsum=0
            for i in range(TOP_K):
                if rnd<cumsum+topk_probs[i]:
                    next_id=topk_ids[i].item()
                    break
                cumsum+=topk_probs[i]

        if next_id in eos_ids+pad_ids+im_end_ids:
            break
        ids=ids+[next_id]
    return tokenizer.decode(ids[1:])
        
    
if __name__=='__main__':
    while True:
        query=input('>')
        if query=='exit':
            break
        
        resp=chat(query)
        print('<',resp)
