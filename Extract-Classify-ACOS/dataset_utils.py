#coding=utf-8

import codecs as cs
import os
import sys
import pdb
sys.path.insert(0, '/mnt/nfs-storage-titan/BERT/pytorch_pretrained_BERT')
from pytorch_pretrained_bert.tokenization import BertTokenizer

def read_pair_gold(f, args):
    # key: text + aspect span + opinion span; value: corresponding category-sentiment type number

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    quad_text = []
    quad_gold = []
    for line in f:
        cur_quad_gold = [[]]
        line = line.strip().split('\t')
        text = line[0].split('####')[0]
        text = text.split(' ')
        cur_text = tokenizer.convert_tokens_to_ids(text)
        # while len(cur_text) < args.max_seq_length:
        #     cur_text.append(0)
        quad_text.append(cur_text)
        cur_quad_gold[0].append(line[0].split('####')[1])
        for ele in line[1:]:
            if ele not in cur_quad_gold[0]:
                cur_quad_gold[0].append(ele)
        quad_gold += cur_quad_gold
    return quad_text, quad_gold


def read_triplet_gold(f, args):
    # key: text + aspect span + opinion span + sentiment type; value: corresponding category type number
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    quad_text = []
    quad_gold = []
    for line in f:
        cur_quad_gold = [[]]
        line = line.strip().split('\t')
        text = line[0].split('####')[0]
        text = text.split(' ')
        cur_text = tokenizer.convert_tokens_to_ids(text)
        # while len(cur_text) < args.max_seq_length:
        #     cur_text.append(0)
        quad_text.append(cur_text)
        cur_quad_gold[0].append(line[0].split('####')[1])
        for ele in line[1:]:
            if ele not in cur_quad_gold[0]:
                cur_quad_gold[0].append(ele)
        quad_gold += cur_quad_gold

    return quad_text, quad_gold