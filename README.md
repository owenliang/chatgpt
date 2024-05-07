# chatgpt

simple decoder-only chat model in pytorch

## 依赖

```
pip install tqdm torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -i https://mirrors.aliyun.com/pypi/simple/
```

## 数据集

放置到dataset目录

```
(chatgpt) (base) owenliang@ubuntu20:~/vscode/chatgpt$ ll dataset/
总用量 2071200
drwxrwxr-x 2 owenliang owenliang       4096 5月   7 20:53 ./
drwxrwxr-x 5 owenliang owenliang       4096 5月   7 21:28 ../
-rw-r--r-- 1 owenliang owenliang   64345976 2月  18  2019 web_text_zh_testa.json
-rw-rw-r-- 1 owenliang owenliang 1991751256 5月   7 20:42 web_text_zh_train.json
-rw-r--r-- 1 owenliang owenliang   64794612 2月  18  2019 web_text_zh_valid.json
```