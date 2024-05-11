from dataset import WebQADataset
import pickle
import os 
import sys 

filename='dataset.bin'

def load_dataset():
    with open(filename,'rb') as fp:
        ds=pickle.load(fp)
        return ds

if __name__=='__main__':
    if os.path.exists(filename):
        ds=load_dataset()
        print(f'{filename}已存在')
        print('训练集大小:',len(ds))
        ids,chatml=ds[5]
        print(ids,chatml)
        sys.exit(0)

    ds=WebQADataset()
    with open(filename,'wb') as fp:
        pickle.dump(ds,fp)
    print('dataset.bin已生成')