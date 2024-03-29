MAX_VOCAB = 10000 # Change back to 50000
try:
    vocab = eval(open('../Dataset/vocab.txt').read())
    VOCAB_SIZE = min(len(vocab), MAX_VOCAB)
except:
    VOCAB_SIZE = MAX_VOCAB

MAX_LEN = 100
NUM_FILES = 5037

MAX_LEN_COMMIT = 15291
MAX_LEN_ISSUE = 495
MAX_LEN_COMMENT = 412636
MAX_NUM_NODES = 13803

AVG_LEN_COMMIT = 68
AVG_LEN_ISSUE = 44
AVG_LEN_COMMENT = 209
AVG_NUM_NODES = 69

COMMIT_LEN = AVG_LEN_COMMIT
COMMENT_LEN = AVG_LEN_COMMENT
ISSUE_LEN = MAX_LEN_ISSUE

HIDDEN_DIM = 128
EMBED_DIM = 128 # Ensure same as hidden_dim
NUM_LAYERS = 2
BATCH_SIZE = 2
EPOCHS=50   
N_GRAPHS = 1
N_COMMITS = 10
TRAIN_SIZE = round(0.7 * NUM_FILES)
VALID_SIZE = round(0.1 * NUM_FILES)
TEST_SIZE = NUM_FILES - TRAIN_SIZE - VALID_SIZE

# Ensure NODE_HID_DIM * GRAPH_N_NODES < HIDDEN_DIM
GRAPH_N_NODES = 70
NODE_HID_DIM = 2

GRAPH_HID_DIM = 2

COMMIT_HID_DIM = 2

# Email: # Change to receive email notifications
EMAIL = 'cs19b031@iittp.ac.in'
