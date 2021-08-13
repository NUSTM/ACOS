#coding=utf-8

import codecs as cs
import pdb

f = cs.open('/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/QuadExtraction-master/fix_data/pair/rest_2016_train_quad_pair.tsv', 'r').readlines()
wf_train = cs.open('/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/Extract-Classify4Quad-2nd/data/rest16_train_pair_syn.tsv', 'w')
wf_dev = cs.open('/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/Extract-Classify4Quad-2nd/data/rest16_dev_pair_syn.tsv', 'w')


train_dict_f = cs.open('/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/Extract-Classify4Quad/data/rest16_train_quad_bert.tsv', 'r').readlines()

# f = cs.open('/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/QuadrupleExtraction/data/laptop/laptop_Entity_Attribute/pair/laptop_train_quad_pair.tsv', 'r').readlines()
# wf_train = cs.open('/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/Extract-Classify4Quad-2nd/data/laptop_train_pair_syn.tsv', 'w')
# wf_dev = cs.open('/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/Extract-Classify4Quad-2nd/data/laptop_dev_pair_syn.tsv', 'w')


# train_dict_f = cs.open('/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/Extract-Classify4Quad/data/laptop_train_quad_bert.tsv', 'r').readlines()

train_dict = {}
for line in train_dict_f:
    line = line.strip().split('\t')
    if line[0] not in train_dict:
        train_dict[line[0]] = 1

pair_dict_train = {}
pair_dict_dev = {}

cnt = 0
for line in f:
    line = line.strip().split('\t')
    text = line[0]
    cate = line[1].split(' ')[0]; senti = line[1].split(' ')[1]
    for ele in line[2:]:
        text_pair = text+'####'+ele
        if text in train_dict:
            if text_pair not in pair_dict_train:
                pair_dict_train[text_pair] = [cnt]
            pair_dict_train[text_pair].append(cate+'#'+senti)
        else:
            if text_pair not in pair_dict_dev:
                pair_dict_dev[text_pair] = [cnt]
            pair_dict_dev[text_pair].append(cate+'#'+senti)
    cnt += 1

list_train = sorted(pair_dict_train.items(),key=lambda x:x[1][0])
list_dev = sorted(pair_dict_dev.items(),key=lambda x:x[1][0])

for ele in list_train:
    wf_train.write(ele[0]+'\t'+' '.join(item for item in ele[1][1:])+'\n')
for ele in list_dev:
    wf_dev.write(ele[0]+'\t'+' '.join(item for item in ele[1][1:])+'\n')