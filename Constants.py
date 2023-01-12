MAX_VOCAB = 50000
try:
    vocab = eval(open('../Dataset/vocab.txt').read())
    VOCAB_SIZE = min(len(vocab), MAX_VOCAB)
except:
    VOCAB_SIZE = MAX_VOCAB
MAX_LEN = 100
HIDDEN_DIM = 128
EMBED_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 8
EPOCHS=50
N_GRAPHS = 1
N_COMMITS = 10
TRAIN_SIZE = 14029 # 70%
VALID_SIZE = 2000 # 10%
TEST_SIZE = 4000 # 20%

