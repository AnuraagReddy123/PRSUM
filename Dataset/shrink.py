import json
import sys

'''
Shrinks the size of the dataset to the specified size. 
'''


size = int(sys.argv[1])

dataset_all = json.load(open('dataset_all.json'))

print('a')

dataset_sh = {}

keys_all = list(dataset_all.keys())
keys_sh = keys_all[:size]

print('a')

for k in keys_sh:
    dataset_sh[k] = dataset_all[k]

print('a')

json.dump(dataset_sh, open('dataset.json', 'w+'))

print('a')
