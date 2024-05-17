# chatgpt

simple decoder-only GTP model in pytorch

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
训练集大小: 258
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [19:04<00:00,  8.74it/s, loss=0.0698]
```

# 推理

```
python chat.py
>山色江声共寂寥
< 山色江声共寂寥，十三陵树晚萧萧
中原事业如江左，芳草何须怨六朝
>三眠
< 三眠未歇，乍到秋时节
一树料阳蝉更咽，曾绾灞陵离别
絮己为萍风卷叶，空凄切
长条莫轻折，苏小恨，倩他说
尽飘零、游冶章台客
红板桥空，湔裙人去，依旧晓风残月
```