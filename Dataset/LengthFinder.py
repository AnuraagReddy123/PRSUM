import json
import math
import matplotlib.pyplot as plt

prs = json.load(open('dataset_filtered.json', 'r'))
len_cm = 0
len_comm = 0
len_issue = 0
cnt = 0
cntissue = 0

maxlen_cm = -math.inf
maxlen_comm = -math.inf
maxlen_issue = -math.inf

for pr in prs.values():
    len_issue += len(pr['issue_title'])
    cntissue += 1

    if len(pr['issue_title']) > maxlen_issue:
        maxlen_issue = len(pr['issue_title'])

    for c in pr['commits'].values():
        len_cm += len(c['cm'])
        len_comm += len(c['comments'])
        cnt += 1

        if len(c['cm']) > maxlen_cm:
            maxlen_cm = len(c['cm'])

        if len(c['comments']) > maxlen_comm:
            maxlen_comm = len(c['comments'])

print('Average length of commit message: ', len_cm / cnt)
print('Average length of commit comments: ', len_comm / cnt)
print('Max length of commit message: ', maxlen_cm)
print('Max length of commit comments: ', maxlen_comm)

