VOCAB_SIZE=500    # 词表大小
MAX_SEQ_LEN=2000     # GPT模型输入限制

# transformer
GPT_DIM=384
GPT_HEAD=6
GPT_FF=1024
GPT_BLOCKS=6

# training
TRAIN_ITER=10000
BATCH_SIZE=50

# inference
TEMPERATURE=1.2
TOP_K=20

# chatml 
SEP='<|sep|>'
# sequence begin & end
BOS='<|beginoftext|>'
EOS='<|endoftext|>'
# padding
PAD='<|padding|>'