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

def measureQuad(pred, gold):
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
    
    res = measureQuad(preds, golds)
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


def getTextType(gold):
    text_type = {}
    for text in gold:
        if text not in text_type:
            text_type[text] = []

        for ele in gold[text]:
            if 4 not in text_type[text]:
                text_type[text].append(4)
            if '-1' not in ele[2] and '-1' not in ele[3]:
                if 0 not in text_type[text]:
                    text_type[text].append(0)
            elif '-1' in ele[2] and '-1' not in ele[3]:
                if 1 not in text_type[text]:
                    text_type[text].append(1)
            elif '-1' not in ele[2] and '-1' in ele[3]:
                if 2 not in text_type[text]:
                    text_type[text].append(2)
            elif '-1' in ele[2] and '-1' in ele[3]:
                if 3 not in text_type[text]:
                    text_type[text].append(3)
    
    return text_type

def measureQuad_imp(pred, gold, text_type):
    tp = [.0, .0, .0, .0, .0]
    fp = [.0, .0, .0, .0, .0]
    fn = [.0, .0, .0, .0, .0]

    # text_set = set()
    # for text in gold:
    #     text_set.add(text)
    # for text in text_set:
    #     for dt in text_type[text]:
    #         cnt = 0
    #         for ele in pred[text]:
    #             if ele in gold[text]:
    #                 cnt += 1
    #         tp[dt] += cnt
    #         fp[dt] += len(pred[text])-cnt

    #         for ele in gold[text]:
    #             if ele not in pred[text]:
    #                 fn[dt] += 1

    for text in pred:
        for dt in text_type[text]:
            cnt = 0
            if text in gold:
                for pair in pred[text]:
                    if pair in gold[text]:
                        cnt += 1
            tp[dt] += cnt
            fp[dt] += len(pred[text])-cnt
            if text in gold:
                fn[dt] += len(gold[text])-cnt
    for text in gold:
        for dt in text_type[text]:
            if text not in pred:
                fn[dt] += len(gold[text])

    for i in range(5):
        print("tp: {}. fp: {}. fn: {}.".format(tp[i], fp[i], fn[i]))
        p = 0 if tp[i] + fp[i] == 0 else 1.*tp[i] / (tp[i] + fp[i])
        r = 0 if tp[i] + fn[i] == 0 else 1.*tp[i] / (tp[i] + fn[i])
        f = 0 if p + r == 0 else 2 * p * r / (p + r)
        print(i, ': ', {'precision':p, 'recall':r, 'micro-F1':f})
    return {'precision':p, 'recall':r, 'micro-F1':f}

def pair_eval(_e, args, logger, tokenizer, model, dataloader, gold, label_list, device, task_name, eval_type='valid'):
    preds = {}
    golds = {}
    quad_preds = {}
    quad_golds = {}
    ids_to_token = {}
    catesenti_dict = {i: label for i, label in enumerate(label_list[0])}
    input_text, quadgold = gold
    for index, cur_quad in enumerate(quadgold):
        cur_input = ' '.join(str(ele) for ele in input_text[index])
        cur_input = cur_input+' '+cur_quad[0]
        golds[cur_input] = cur_quad[1:]
        ori_text = ' '.join(ele for ele in tokenizer.convert_ids_to_tokens(input_text[index]))
        ids_to_token[cur_input] = ori_text+' '+cur_quad[0]

        quad_pairs = []
        for ele in cur_quad[1:]:
            ele = ele.split('#')
            cate = '#'.join(item for item in ele[:-1]); senti = ele[-1]
            asp = cur_quad[0].split(' ')[0]; opi = cur_quad[0].split(' ')[1]
            tmp_quad = [cate, senti, asp, opi]
            if tmp_quad not in quad_pairs:
                quad_pairs.append(tmp_quad)
        if ori_text in quad_golds:
            quad_golds[ori_text] += quad_pairs
        else:
            quad_golds[ori_text] = quad_pairs
    tmp_cnt = 0
    for step, batch in enumerate(dataloader):

        batch = tuple(t.to(device) for t in batch)
        _tokens_len, _aspect_input_ids, _aspect_input_mask, _aspect_segment_ids, _candidate_aspect, \
        _candidate_opinion, _label_id = batch

        # define a new function to compute loss values for both output_modes
        with torch.no_grad():
            loss, logits = model(tokenizer, _e, aspect_input_ids=_aspect_input_ids,
                    aspect_token_type_ids=_aspect_segment_ids, aspect_attention_mask=_aspect_input_mask,
                    candidate_aspect=_candidate_aspect, candidate_opinion=_candidate_opinion, label_id=_label_id)

        logits = logits[0].detach().cpu().numpy()
        # pair_matrix = logits[0].view(len(_tokens_len), logits[1].item(), logits[1].item(), 3).detach().cpu().numpy()

        for i in range(len(_tokens_len)):
            #得到输入文本作为key，相应的类别预测结果作为value
            aspect_len = _aspect_input_mask[i].detach().cpu().numpy().sum()
            aspect_tags = _candidate_aspect[i].detach().cpu().numpy()
            opinion_tags = _candidate_opinion[i].detach().cpu().numpy()
            entity_label = r'11*'

            aspect_labels = ''.join(str(ele) for ele in aspect_tags)
            cur_aspect = []
            for ele in re.finditer(entity_label, aspect_labels):
                # if (ele.end()-ele.start())<_tokens_len[i]-2:
                if ele.start() == 0 and '-1,-1' not in cur_aspect:
                    cur_aspect.append('-1,-1')
                elif (ele.start() > 0 and ele.end()<aspect_len):
                    cur_aspect.append(str(ele.start()-1) + ',' + str(ele.end()-1))

            opinion_labels = ''.join(str(ele) for ele in opinion_tags)
            cur_opinion = []
            for ele in re.finditer(entity_label, opinion_labels):
                if ele.start() == (aspect_len-1) and '-1,-1' not in cur_opinion:
                    cur_opinion.append('-1,-1')
                elif (ele.start() > 0 and ele.end()<aspect_len):
                    cur_opinion.append(str(ele.start()-1) + ',' + str(ele.end()-1))

            if len(cur_aspect) == 1 and len(cur_opinion) == 1:
                cur_ao = cur_aspect[0]+' '+cur_opinion[0]
                pred_res = []
                ind = np.where(logits[i]>0)
                for ele in ind[0]:
                    pred_res.append(catesenti_dict[int(ele)])
                ttt = (_aspect_input_ids[i].detach().cpu().numpy().tolist())[1:(_tokens_len[i]-1)]
                cur_input = ' '.join(str(ele) for ele in ttt)+' '+cur_ao
                ids_to_token[cur_input] = ' '.join(ele for ele in tokenizer.convert_ids_to_tokens(ttt))+' '+cur_ao
                
                preds[cur_input] = pred_res
                
                quad_pairs = []
                for ele in pred_res:
                    ele = ele.split('#')
                    cate = '#'.join(item for item in ele[:-1]); senti = ele[-1]
                    tmp_quad = [cate, senti, cur_aspect[0], cur_opinion[0]]
                    if tmp_quad not in quad_pairs:
                        quad_pairs.append(tmp_quad)
                tmp_cnt += len(quad_pairs)
                ori_text = ' '.join(ele for ele in tokenizer.convert_ids_to_tokens(ttt))
                if ori_text in quad_preds:
                    quad_preds[ori_text] += quad_pairs
                else:
                    quad_preds[ori_text] = quad_pairs
    print("Quad num: {}".format(tmp_cnt))
    # pdb.set_trace()
    res = measureQuad(preds, golds)
    text_type = getTextType(quad_golds)
    # tmp = measureQuad_imp(quad_preds, quad_golds)

    if eval_type == 'valid':
        logger.info("***** Eval results *****")
        for key in sorted(res.keys()):
            logger.info("  %s = %s", key, str(res[key]))
        return res

    elif eval_type == 'test':
        
        # evaluation for all sub-tasks, we do quad extraction, so the element number is 4.
        ele_num = 4
        index_to_name = {0:'category', 1:'sentiment', 2:'aspect', 3:'opinion'}
        for comb_choice in range(1, (1<<ele_num)):
            exist_index = []
            cnt = 0
            while comb_choice:
                if comb_choice&1:
                    exist_index.append(cnt)
                cnt += 1
                comb_choice >>= 1
            sub_preds = {}
            sub_golds = {}
            for cur_key in quad_preds:
                cur_subs = []
                for quad in quad_preds[cur_key]:
                    cur_sub = [quad[index] for index in exist_index]
                    if cur_sub not in cur_subs:
                        cur_subs.append(cur_sub)
                sub_preds[cur_key] = cur_subs
            for cur_key in quad_golds:
                cur_subs = []
                for quad in quad_golds[cur_key]:
                    cur_sub = [quad[index] for index in exist_index]
                    if cur_sub not in cur_subs:
                        cur_subs.append(cur_sub)
                sub_golds[cur_key] = cur_subs
            sub_res = measureQuad_imp(sub_preds, sub_golds, text_type)
            subtask_name = ' '.join(index_to_name[ele] for ele in exist_index)
            # if subtask_name == 'aspect':
            #     pdb.set_trace()
            logger.info("***** %s results *****", subtask_name)
            for key in sorted(sub_res.keys()):
                logger.info("  {} = {:.2%}".format(key, sub_res[key]))
            logger.info("-----------------------------------")

        pipeline_res = cs.open(args.output_dir+os.sep+'result.txt', 'w')
        for key in golds:
            pipeline_res.write(ids_to_token[key]+'\n')
            for cur_pair in golds[key]:
                pipeline_res.write(cur_pair+'\t')
            pipeline_res.write('\n')
            if key in preds:
                for cur_pair in preds[key]:
                    pipeline_res.write(cur_pair+'\t')
            pipeline_res.write('\n\n')
        for key in preds:
            if key not in golds:
                pipeline_res.write(ids_to_token[key]+'\n')
                pipeline_res.write('\n')
                for cur_pair in preds[key]:
                    pipeline_res.write(cur_pair+'\t')
                pipeline_res.write('\n\n')

        logger.info("***** Test results *****")
        for key in sorted(res.keys()):
            logger.info("  %s = %s", key, str(res[key]))
        return res