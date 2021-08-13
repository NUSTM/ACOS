#coding=utf-8
import random
import numpy as np
import codecs as cs
from sklearn.model_selection import KFold

seed = 42
random.seed(seed)
np.random.seed(seed)

kf = KFold(n_splits=10, shuffle=True, random_state=seed)

for fold_name in ['rest16', 'laptop']:
    f = cs.open(fold_name + '_train_quad_bert.tsv', 'r').readlines()

    wf_t = cs.open(fold_name + '_train_quad_bert_n.tsv', 'w')
    wf_d = cs.open(fold_name + '_dev_quad_bert_n.tsv', 'w')

    ori = []
    for line in f:
        line = line.strip()
        ori.append(line)
    ori = np.array(ori)

    for fold, (train_index, valid_index) in enumerate(kf.split(ori)):
        t_ori = ori[train_index]
        d_ori = ori[valid_index]
        for ele in t_ori:
            wf_t.write(ele+'\n')
        for ele in d_ori:
            wf_d.write(ele+'\n')
        break