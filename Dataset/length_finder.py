import json
import math
import matplotlib.pyplot as plt
import os

prs = json.load(open('dataset_filtered.json', 'r'))
len_cm = 0
len_comm = 0
len_issue = 0
num_nodes = 0
cnt = 0
cntissue = 0
cntgraphs = 0

maxlen_cm = -math.inf
maxlen_comm = -math.inf
maxlen_issue = -math.inf
max_nodes = -math.inf
max_pr = None

num_files = len(os.listdir('data'))

for pr in prs.values():
    # len_issue += len(pr['issue_titles'])
    # cntissue += 1

    # if len(pr['issue_titles']) > maxlen_issue:
    #     maxlen_issue = len(pr['issue_titles'])

    for c in pr['commits'].values():
        len_cm += len(c['cm'])
        len_comm += len(c['comments'])
        cnt += 1

        if len(c['cm']) > maxlen_cm:
            maxlen_cm = len(c['cm'])

        if len(c['comments']) > maxlen_comm:
            maxlen_comm = len(c['comments'])
            max_pr = pr

print('Average length of commit message: ', len_cm // cnt)
print('Average length of commit comments: ', len_comm // cnt)
print('Max length of commit message: ', maxlen_cm)
print('Max length of commit comments: ', maxlen_comm)
print('Pr: max comment', max_pr['id'])

# Replace maxlen of commit messages and comments in Constants.py file
# Replace maxlen of issue titles in Constants.py

files = os.listdir('data')
for file in files:
    f = json.load(open(f'data/{file}', 'r'))
    len_issue += len(f['issue_title'])
    cntissue += 1
    if len(f['issue_title']) > maxlen_issue:
        maxlen_issue = len(f['issue_title'])
    
    for commit in f['commits'].values():
        for graph in commit['graphs']:
            num_nodes += len(graph['node_features'])
            cntgraphs += 1
            if len(graph['node_features']) > max_nodes:
                max_nodes = len(graph['node_features'])


constfile = open('../Constants_dup.py', 'w+')
for line in open('../Constants.py', 'r'):
    if line.startswith('MAX_LEN_COMMIT'):
        constfile.write(f'MAX_LEN_COMMIT = {maxlen_cm}\n')
    elif line.startswith('MAX_LEN_COMMENT'):
        constfile.write(f'MAX_LEN_COMMENT = {maxlen_comm}\n')
    elif line.startswith('AVG_LEN_COMMIT'):
        constfile.write(f'AVG_LEN_COMMIT = {len_cm // cnt}\n')
    elif line.startswith('AVG_LEN_COMMENT'):
        constfile.write(f'AVG_LEN_COMMENT = {len_comm // cnt}\n')
    elif line.startswith('MAX_LEN_ISSUE'):
        constfile.write(f'MAX_LEN_ISSUE = {maxlen_issue}\n')
    elif line.startswith('AVG_LEN_ISSUE'):
        constfile.write(f'AVG_LEN_ISSUE = {len_issue // cntissue}\n')
    elif line.startswith('MAX_NUM_NODES'):
        constfile.write(f'MAX_NUM_NODES = {max_nodes}\n')
    elif line.startswith('AVG_NUM_NODES'):
        constfile.write(f'AVG_NUM_NODES = {num_nodes // cntgraphs}\n')
    elif line.startswith('NUM_FILES'):
        constfile.write(f'NUM_FILES = {num_files}\n')
    else:
        constfile.write(line)

constfile.close()
# Delete Constants.py and rename Constants_dup.py to Constants.py
os.remove('../Constants.py')
os.rename('../Constants_dup.py', '../Constants.py')