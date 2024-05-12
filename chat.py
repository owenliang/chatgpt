from gpt import GPT
from config import *
import torch 
from bpe import BPETokenizer
import torch.nn.functional as F
from build_dataset import load_dataset

# 设备
DEVICE='cuda' if torch.cuda.is_available() else 'cpu' 

# 分词器
tokenizer=BPETokenizer()  
tokenizer.load('tokenizer.bin')
tokenizer.add_special_tokens([IM_START,IM_END,BOS,EOS,PAD])

# 加载模型
model=GPT(d_model=GPT_DIM,nhead=GPT_HEAD,feedforward=GPT_FF,vocab_size=tokenizer.vocab_size(),seq_max_len=MAX_SEQ_LEN).to(DEVICE) # 模型
try:  
    model.load_state_dict(torch.load('model.pth'))
except:
    pass

# 可能的结束符
im_end_ids,_=tokenizer.encode(IM_END)
eos_ids,_=tokenizer.encode(EOS)
pad_ids,_=tokenizer.encode(PAD)

def chat(query):
    global tokenizer,model
    
    resp_ids=[] # assitant回答

    prompt=f"{BOS}{IM_START}system\n你是聪明的个人助理\n{IM_END}\n{IM_START}user\n{query}\n{IM_END}\n{IM_START}assitant\n"
    ids,_=tokenizer.encode(prompt) 
    
    while len(ids)<MAX_SEQ_LEN:
        batch_ids=torch.tensor([ids],dtype=torch.long).to(DEVICE)
        batch_paddding_mask=torch.tensor([[0]*len(ids)],dtype=torch.bool).to(DEVICE)
        
        logits=model(batch_ids,batch_paddding_mask) # (batch,seq,vocab)
        
        probs=F.softmax(logits,dim=-1)
        next_id=probs[0,-1,:].argmax().item()
        if next_id in im_end_ids+eos_ids+pad_ids:
            break
        resp_ids.append(next_id)
        ids=ids+[next_id]
    return tokenizer.decode(resp_ids)
        
    
if __name__=='__main__':
    while True:
        query=input('>')
        if query=='exit':
            break
        
        resp=chat(query)
        print('<',resp)