# chatgpt

simple decoder-only chat model in pytorch

## 依赖

```
pip install tqdm torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -i https://mirrors.aliyun.com/pypi/simple/
```

## 训练

训练tokenizer

```
python train_tokenizer.py
```

构建dataset

```
python build_dataset.py
```

训练gpt

```
python train_gpt.py
```