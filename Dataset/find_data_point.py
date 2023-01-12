'''
Finds the particular datapoint requested
'''

import json
import sys

datapoint = sys.argv[1]
dataset = json.load(open('dataset.json', 'r'))
dataset_testing = dict()

for d_key in dataset:
    if d_key == datapoint:
        # Add it into file
        dataset_testing[d_key] = dataset[d_key]
        json.dump(dataset_testing, open('dataset_testing.json', 'w+'))
        print(f'Found datapoint {datapoint}')
        break