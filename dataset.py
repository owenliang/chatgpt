from torch.utils.data import Dataset
from bpe import BPETokenizer
from config import *
import json 

# 问答数据集
class WebQADataset(Dataset):
    def __init__(self):
        super().__init__()
        self.tokenizer=BPETokenizer()
        self.tokenizer.load('tokenizer.bin')
        self._build_train_data()
    
    def _build_train_data(self):
        self.data=[]
        with open('dataset/web_text_zh_train.json','r') as fp:
            for line in fp:
                try:
                    row=json.loads(line.strip())
                    if row['star']<STAR_THERSHOLD: # 保留高质量数据
                        continue
                    chatml=f"{IM_START}system\n你是聪明的个人助理\n{IM_END}\n{IM_START}user\n{row['title']}\n{IM_END}\n{IM_START}assitant\n{row['content']}\n{IM_END}"
                    ids,tokens=self.tokenizer.encode(chatml)
                    if len(ids)>MAX_SEQ_LEN-2:  # 留出BOS和EOS的token
                        continue
                    self.data.append((ids,chatml))
                except:
                    continue 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]
