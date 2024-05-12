STAR_THERSHOLD=1000 # 保留点赞高的数据
VOCAB_SIZE=30000    # 词表大小
MAX_SEQ_LEN=2000     # 最长输入token

# transformer
GPT_DIM=128
GPT_HEAD=4
GPT_FF=256

# training
BATCH_SIZE=5

# chatml 
IM_START='<|im_start|>' 
IM_END='<|im_end|>'
# sequence begin & end
BOS='<|beginoftext|>'
EOS='<|endoftext|>'
# padding
PAD='<|padding|>'