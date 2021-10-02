#coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange
import pdb
import warnings
import codecs as cs
import copy
import re
# warnings.filterwarnings('ignore')

import numpy as np

import torch
from torch.nn import CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss, BCEWithLogitsLoss

from run_classifier_dataset_utils import compute_metrics

def measureTasks(pred, gold):
    tp = .0
    fp = .0
    fn = .0
    for text in pred:
        cnt = 0
        if text in gold:
            for pair in pred[text]:
                if pair in gold[text]:
                    cnt += 1
        tp += cnt
        fp += len(pred[text])-cnt
        if text in gold:
            fn += len(gold[text])-cnt
    for text in gold:
        if text not in pred:
            fn += len(gold[text])

    print("tp: {}. fp: {}. fn: {}.".format(tp, fp, fn))
    
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return {'precision':p, 'recall':r, 'micro-F1':f}

def pred_eval(_e, args, logger, tokenizer, model, dataloader, eval_gold, label_list, device, task_name, eval_type='valid'):
    
    preds = {}
    golds = {}
    ids_to_token = {}
    pred_aspect_tag = []
    pred_imp_aspect = []
    pred_imp_opinion = []
    input_text, pairgold = eval_gold
    _all_tokens_len = []
    input_length_map = {}
    entity_label = r'32*'
    opinion_entity_label = r'54*'
    label_map_seq = {label : i for i, label in enumerate(label_list[1])}

    for index in range(0, len(pairgold), 3):
        cur_quad = pairgold[index]
        gold_imp_aspect = pairgold[index+1]
        gold_imp_opinion = pairgold[index+2]
        gold_tag = []
        cur_aspect_tag = ''.join(str(ele) for ele in cur_quad)
        max_len = len(cur_aspect_tag)
        for ele in re.finditer(entity_label, cur_aspect_tag):
            gold_tag.append('a-' + str(ele.start()) + ',' + str(ele.end()))
        if gold_imp_aspect == 1:
            gold_tag.append('a--1,-1')
    
        for ele in re.finditer(opinion_entity_label, cur_aspect_tag):
            gold_tag.append('o-' + str(ele.start()) + ',' + str(ele.end()))
        if gold_imp_opinion == 1:
            gold_tag.append('o--1,-1')
        
        cur_input = ' '.join(str(ele) for ele in input_text[index//3])
        golds[cur_input] = gold_tag
        ids_to_token[cur_input] = ' '.join(ele for ele in tokenizer.convert_ids_to_tokens(input_text[index//3]))

    for step, batch in enumerate(dataloader):

        if step % 500 == 0 and step>0:
            print(step)

        _all_tokens_len += batch[0].numpy().tolist()
        batch = tuple(t.to(device) for t in batch)
        _tokens_len, _aspect_input_ids, _aspect_input_mask, _aspect_ids, _aspect_segment_ids, \
                _exist_imp_aspect, _exist_imp_opinion = batch

        with torch.no_grad():
            _, logits = model(aspect_input_ids=_aspect_input_ids, aspect_labels=_aspect_ids,
                aspect_token_type_ids=_aspect_segment_ids, aspect_attention_mask=_aspect_input_mask,
                exist_imp_aspect=_exist_imp_aspect, exist_imp_opinion=_exist_imp_opinion)

            # input '[CLS] text [SEP] category/sentiment [SEP]', obtain '[CLS] text' only, first '[SEP]' is used to predict
            # the existence of implicit aspect or opinion.

            logits_imp_aspect = np.argmax(logits[1].detach().cpu().numpy(), axis=-1).tolist()
            logits_imp_opinion = np.argmax(logits[2].detach().cpu().numpy(), axis=-1).tolist()
            for i, ele in enumerate(logits[0]):
                pred_aspect_tag.append(ele)
            for i, ele in enumerate(logits_imp_aspect):
                pred_imp_aspect.append(ele)
            for i, ele in enumerate(logits_imp_opinion):
                pred_imp_opinion.append(ele)

    for i in range(len(pred_aspect_tag)):
        cur_aspect_tag = ''.join(str(ele) for ele in pred_aspect_tag[i])
        pred_tag = []
        for ele in re.finditer(entity_label, cur_aspect_tag):
            pred_tag.append('a-'+str(ele.start()-1) + ',' + str(ele.end()-1))
        if pred_imp_aspect[i] == 1:
            pred_tag.append('a--1,-1')
        
        for ele in re.finditer(opinion_entity_label, cur_aspect_tag):
            pred_tag.append('o-'+str(ele.start()-1) + ',' + str(ele.end()-1))
        if pred_imp_opinion[i] == 1:
            pred_tag.append('o--1,-1')

        cur_input = ' '.join(str(ele) for ele in input_text[i])
        preds[cur_input] = pred_tag
        input_length_map[cur_input] = _all_tokens_len[i]
    
    res = measureTasks(preds, golds)
    if eval_type == 'valid':
        pipeline_file = cs.open(args.output_dir+os.sep+'valid.txt', 'w')
    else:
        pipeline_file = cs.open(args.output_dir+os.sep+'pred4pipeline.txt', 'w')
    for text in preds:
        length = input_length_map[text]-1
        cur_text = ids_to_token[text]
        cur_text = cur_text.split(' ')[1:length]
        if len(preds[text]) > 0:
            pipeline_file.write(' '.join(ele for ele in cur_text)+'\t'+'\t'.join(ele for ele in preds[text])+'\n')

    if eval_type == 'valid':
        logger.info("***** Eval results *****")
        for key in sorted(res.keys()):
            logger.info("  %s = %s", key, str(res[key]))
        return res

    elif eval_type == 'test':
        logger.info("***** Test results *****")
        for key in sorted(res.keys()):
            logger.info("  %s = %s", key, str(res[key]))
        return res