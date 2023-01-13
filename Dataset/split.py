import sys
sys.path.append('.')
sys.path.append('..')

import random
import Constants


filenames = open('data.txt').readlines()
random.shuffle(filenames)

tr = Constants.TRAIN_SIZE
vl = Constants.VALID_SIZE
ts = Constants.TEST_SIZE
fns_train = filenames[:tr]
fns_valid = filenames[tr:tr+vl]
fns_test = filenames[tr+vl:tr+vl+ts]

open('data_train.txt', 'w+').write(''.join(fns_train))
open('data_valid.txt', 'w+').write(''.join(fns_valid))
open('data_test.txt', 'w+').write(''.join(fns_test))
