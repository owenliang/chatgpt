from torch.utils.data import Dataset
from tokenizer import BPETokenizer
from config import *
import json 
from tqdm import tqdm

# 数据集
class NalanDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open('纳兰性德诗集.json','r',encoding='utf-8') as fp:
            self.raw_ds=json.loads(fp.read())
    
    def build_train_data(self):
        tokenizer=BPETokenizer()
        tokenizer.load('tokenizer.bin')
        
        self.data=[]
        for sample in tqdm(self.raw_ds,desc='building dataset'):
            try:
                text='\n'.join(sample['para'])
                ids,_=tokenizer.encode(text)
                if len(ids)>MAX_SEQ_LEN-2:  # 留出BOS和EOS的token
                    continue
                self.data.append((ids,text))
            except:
                continue
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]