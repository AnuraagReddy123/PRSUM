'''
Counts the number of datapoints in the given dataset
'''

import json
import sys

name = sys.argv[1]

dataset = json.load(open(name))

print(len(dataset))