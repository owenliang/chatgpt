import torch 
from build_dataset import load_dataset
from gpt import GPT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from config import *
from bpe import BPETokenizer

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

dataset=load_dataset() # 数据集
print('训练集大小:',len(dataset))

tokenizer=BPETokenizer()  # 分词器
tokenizer.load('tokenizer.bin')
tokenizer.add_special_tokens([IM_START,IM_END,BOS,EOS,PAD])

model=GPT(d_model=GPT_DIM,nhead=GPT_HEAD,feedforward=GPT_FF,vocab_size=tokenizer.vocab_size(),seq_max_len=MAX_SEQ_LEN).to(DEVICE) # 模型
try:    # 加载模型
    model.load_state_dict(torch.load('model.pth'))
except:
    pass 

optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)   # 优化器

def batch_proc(batch):
    bos_ids,_=tokenizer.encode(BOS)
    eos_ids,_=tokenizer.encode(EOS)
    pad_ids,_=tokenizer.encode(PAD)
    
    batch_x=[]
    batch_chatml=[]
    # bpe encode
    for sample in batch:
        ids,chatml=sample
        ids=bos_ids+ids+eos_ids
        batch_x.append(ids)
        batch_chatml.append(chatml)
    
    # padding
    max_len=max([len(ids) for ids in batch_x])
    for ids in batch_x:
        if len(ids)<max_len:
            ids.extend(pad_ids*(max_len-len(ids)))
    batch_x=torch.tensor(batch_x,dtype=torch.long)
    
    # padding mask
    batch_padding_mask=(batch_x==pad_ids[0])
    return batch_x,batch_padding_mask,batch_chatml
    
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True,collate_fn=batch_proc)    # 数据加载器

'''
    训练模型
'''

pad_ids,_=tokenizer.encode(PAD)

EPOCH=500

iter_count=0
for epoch in range(EPOCH):
    for batch_ids,batch_padding_mask,batch_chatml in dataloader:
        batch_ids=batch_ids.to(DEVICE)
        batch_padding_mask=batch_padding_mask.to(DEVICE)
        
        logtis=model(batch_ids,batch_padding_mask)  # (batch,seq,vocab)
        
        probs=logtis[:,:-1,:]   # (batch,seq-1,vocab)
        targets=batch_ids[:,1:] # (batch,seq-1)
        loss=F.cross_entropy(probs.reshape(-1,probs.size(2)),targets.reshape(-1),ignore_index=pad_ids[0])

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        if iter_count%1000==0:
            print('epoch:{} iter:{},loss:{}'.format(epoch,iter_count,loss))
            torch.save(model.state_dict(),'.model.pth')
            os.replace('.model.pth','model.pth')
        iter_count+=1