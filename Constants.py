MAX_VOCAB = 50000
try:
    vocab = eval(open('../Dataset/vocab.txt').read())
    VOCAB_SIZE = min(len(vocab), MAX_VOCAB)
except:
    VOCAB_SIZE = MAX_VOCAB

MAX_LEN = 100
COMMIT_LEN = 70 # Set 0 to disable
COMMENT_LEN = 200 # Set 0 to disable
ISSUE_LEN = 100 # Set 0 to disable

HIDDEN_DIM = 128
EMBED_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 4
EPOCHS=50
N_GRAPHS = 3
N_COMMITS = 10
TRAIN_SIZE = 914 # 70%
VALID_SIZE = 130 # 10%
TEST_SIZE = 261 # 20%

# Ensure NODE_HID_DIM * GRAPH_N_NODES < HIDDEN_DIM
GRAPH_N_NODES = 70
NODE_HID_DIM = 2

GRAPH_HID_DIM = 2

COMMIT_HID_DIM = 2

# Email: # Change to receive email notifications
EMAIL = 'cs19b031@iittp.ac.in'
