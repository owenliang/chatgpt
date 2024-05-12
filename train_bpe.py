from bpe import BPETokenizer
from config import *
import json 
import os
import sys 

if os.path.exists('tokenizer.bin'):
    print('tokenizer.bin已存在')
    sys.exit(0)

# 加载语料
text_list=[]
sample_count=0
with open('dataset/web_text_zh_train.json','r') as fp:
    for line in fp:
        try:
            row=json.loads(line.strip())
            if row['star']<STAR_THERSHOLD:  # bpe实在太慢，数据太多练不动，只保留少量优质回答做分词训练
                continue
            text_list.append(row['title'])
            text_list.append(row['content'])
        except:
            continue 
        sample_count+=1
print('共加载%d条数据'%sample_count)

# 训练词表
tokenizer=BPETokenizer()  
tokenizer.train(text_list,VOCAB_SIZE)
tokenizer.add_special_tokens([IM_START,IM_END,BOS,EOS,PAD])
tokenizer.save('tokenizer.bin')