# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange
import pdb
from collections import defaultdict, namedtuple
from manager import *
import math
import codecs as cs
from sklearn.model_selection import KFold

gm = GPUManager()
device = gm.auto_choice(mode=0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss, BCEWithLogitsLoss

from modeling import BertForQuadABSA
from bert_utils.tokenization import BertTokenizer
from bert_utils.optimization import BertAdam, WarmupLinearSchedule

from run_classifier_dataset_utils import *
from eval_metrics import *
import gc

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input source data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--domain_type",
                        default=None,
                        type=str,
                        required=True,
                        help="domain to choose.")

    parser.add_argument("--model_type",
                        default=None,
                        type=str,
                        required=True,
                        help="model to choose.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # print(args.output_dir)
    # pdb.set_trace()

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels(args.domain_type)
    num_labels = len(label_list[1])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model_dict = {
        'quad': BertForQuadABSA,
    }

    label_map_senti = {label : i for i, label in enumerate(label_list[0])}
    label_map_seq = {label : i for i, label in enumerate(label_list[1])}

    global_step = 0
    nb_tr_steps = 0
    eval_gold = []
    valid_gold = []
    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir, args.domain_type)
        f = cs.open(args.data_dir+'/tokenized_data/'+args.domain_type +'_test_quad_bert.tsv', 'r').readlines()
        for line in f:
            cur_exist_imp_aspect = 0
            cur_exist_imp_opinion = 0
            line = line.strip().split('\t')
            cur_text = line[0]
            aspect_labels = [label_map_seq['O'] for ele in range(args.max_seq_length)]
            
            for quad in line[1:]:
                cur_aspect = quad.split(' ')[0]; cur_opinion = quad.split(' ')[-1]
                a_st = int(cur_aspect.split(',')[0]); a_ed = int(cur_aspect.split(',')[1])
                
                if a_ed != -1:
                    aspect_labels[a_st] = label_map_seq['B-A']
                    for i in range(a_st+1, a_ed):
                        aspect_labels[i] = label_map_seq['I-A']
                else:
                    cur_exist_imp_aspect = 1

                o_st = int(cur_opinion.split(',')[0]); o_ed = int(cur_opinion.split(',')[1])
                
                if o_ed != -1:
                    aspect_labels[o_st] = label_map_seq['B-O']
                    for i in range(o_st+1, o_ed):
                        aspect_labels[i] = label_map_seq['I-O']
                else:
                    cur_exist_imp_opinion = 1

            eval_gold += [aspect_labels, cur_exist_imp_aspect, cur_exist_imp_opinion]

        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, task_name)

        eval_tokens_len = torch.tensor([f.tokens_len for f in eval_features], dtype=torch.long)
        eval_aspect_input_ids = torch.tensor([f.aspect_input_ids for f in eval_features], dtype=torch.long)
        eval_aspect_input_mask = torch.tensor([f.aspect_input_mask for f in eval_features], dtype=torch.long)
        eval_aspect_ids = torch.tensor([f.aspect_ids for f in eval_features], dtype=torch.long)
        eval_aspect_segment_ids = torch.tensor([f.aspect_segment_ids for f in eval_features], dtype=torch.long)
        eval_exist_imp_aspect = torch.tensor([f.exist_imp_aspect for f in eval_features], dtype=torch.long)
        eval_exist_imp_opinion = torch.tensor([f.exist_imp_opinion for f in eval_features], dtype=torch.long)

        eval_gold = [eval_aspect_input_ids.numpy().tolist(), eval_gold]

        eval_data = TensorDataset(eval_tokens_len, eval_aspect_input_ids, eval_aspect_input_mask, eval_aspect_ids,
        eval_aspect_segment_ids, eval_exist_imp_aspect, eval_exist_imp_opinion)
        # Run prediction for full data
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)  # Note that this sampler samples randomly
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_train:

        # Prepare data loader
        train_examples = processor.get_train_examples(args.data_dir, args.domain_type)

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, task_name)

        tokens_len = torch.tensor([f.tokens_len for f in train_features], dtype=torch.long)
        aspect_input_ids = torch.tensor([f.aspect_input_ids for f in train_features], dtype=torch.long)
        aspect_input_mask = torch.tensor([f.aspect_input_mask for f in train_features], dtype=torch.long)
        aspect_ids = torch.tensor([f.aspect_ids for f in train_features], dtype=torch.long)
        aspect_segment_ids = torch.tensor([f.aspect_segment_ids for f in train_features], dtype=torch.long)
        exist_imp_aspect = torch.tensor([f.exist_imp_aspect for f in train_features], dtype=torch.long)
        exist_imp_opinion = torch.tensor([f.exist_imp_opinion for f in train_features], dtype=torch.long)

        valid_examples = processor.get_valid_examples(args.data_dir, args.domain_type)
        f = cs.open(args.data_dir+'/tokenized_data/'+args.domain_type +'_dev_quad_bert.tsv', 'r').readlines()
        for line in f:
            cur_exist_imp_aspect = 0
            cur_exist_imp_opinion = 0
            line = line.strip().split('\t')
            cur_text = line[0]
            aspect_labels = [label_map_seq['O'] for ele in range(args.max_seq_length)]
            
            for quad in line[1:]:
                cur_aspect = quad.split(' ')[0]; cur_opinion = quad.split(' ')[-1]
                a_st = int(cur_aspect.split(',')[0]); a_ed = int(cur_aspect.split(',')[1])
                
                if a_ed != -1:
                    aspect_labels[a_st] = label_map_seq['B-A']
                    for i in range(a_st+1, a_ed):
                        aspect_labels[i] = label_map_seq['I-A']
                else:
                    cur_exist_imp_aspect = 1
                    
                o_st = int(cur_opinion.split(',')[0]); o_ed = int(cur_opinion.split(',')[1])
                
                if o_ed != -1:
                    aspect_labels[o_st] = label_map_seq['B-O']
                    for i in range(o_st+1, o_ed):
                        aspect_labels[i] = label_map_seq['I-O']
                else:
                    cur_exist_imp_opinion = 1

            valid_gold += [aspect_labels, cur_exist_imp_aspect, cur_exist_imp_opinion]

        valid_features = convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer, output_mode, task_name)

        valid_tokens_len = torch.tensor([f.tokens_len for f in valid_features], dtype=torch.long)
        valid_aspect_input_ids = torch.tensor([f.aspect_input_ids for f in valid_features], dtype=torch.long)
        valid_aspect_input_mask = torch.tensor([f.aspect_input_mask for f in valid_features], dtype=torch.long)
        valid_aspect_ids = torch.tensor([f.aspect_ids for f in valid_features], dtype=torch.long)
        valid_aspect_segment_ids = torch.tensor([f.aspect_segment_ids for f in valid_features], dtype=torch.long)
        valid_exist_imp_aspect = torch.tensor([f.exist_imp_aspect for f in valid_features], dtype=torch.long)
        valid_exist_imp_opinion = torch.tensor([f.exist_imp_opinion for f in valid_features], dtype=torch.long)

        # Prepare optimizer

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)

        all_results = []
            
        model = model_dict[args.model_type].from_pretrained(args.bert_model, num_labels=num_labels)
        if args.local_rank == 0:
            torch.distributed.barrier()

        if args.fp16:
            model.half()

        model.to(device)

        train_data = TensorDataset(tokens_len, aspect_input_ids, aspect_input_mask,
        aspect_ids, aspect_segment_ids, exist_imp_aspect, exist_imp_opinion)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        valid_gold = [valid_aspect_input_ids.numpy().tolist(), valid_gold]

        valid_data = TensorDataset(valid_tokens_len, valid_aspect_input_ids, valid_aspect_input_mask,
        valid_aspect_ids, valid_aspect_segment_ids, valid_exist_imp_aspect, valid_exist_imp_opinion)
        if args.local_rank == -1:
            valid_sampler = SequentialSampler(valid_data)
        else:
            valid_sampler = DistributedSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.eval_batch_size)

        num_train_optimization_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                warmup=args.warmup_proportion,
                                t_total=num_train_optimization_steps)

        max_macro_F1 = -1.0

        for _e in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                _tokens_len, _aspect_input_ids, _aspect_input_mask, _aspect_ids, _aspect_segment_ids, \
                    _exist_imp_aspect, _exist_imp_opinion = batch

                # define a new function to compute loss values for both output_modes

                losses, logits = model(aspect_input_ids=_aspect_input_ids, aspect_labels=_aspect_ids,
                aspect_token_type_ids=_aspect_segment_ids, aspect_attention_mask=_aspect_input_mask,
                exist_imp_aspect=_exist_imp_aspect, exist_imp_opinion=_exist_imp_opinion)

                if step % 30 == 0:
                    logger.info('Total Loss is {} .'.format(losses[0]))
                step += 1
                loss = losses[0]
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                else:
                    loss = loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    ae_loss = ae_loss / args.gradient_accumulation_steps

                if args.fp16:
                    # optimizer.backward(loss)
                    optimizer.backward(ae_loss)
                else:
                    loss.backward()

                nb_tr_examples += _aspect_input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            model.eval()
            result = pred_eval(_e, args, logger, tokenizer, model, valid_dataloader, valid_gold, label_list, device, task_name, eval_type='valid')

            if max_macro_F1 < result['micro-F1']:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                dirs_name = args.output_dir
                if not os.path.exists(dirs_name):
                    os.mkdir(dirs_name)
                output_model_file = os.path.join(dirs_name, WEIGHTS_NAME)
                output_config_file = os.path.join(dirs_name, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(dirs_name)

                final_result = pred_eval(_e, args, logger, tokenizer, model, eval_dataloader, eval_gold, label_list, device, task_name, eval_type='test')
                max_macro_F1 = result['micro-F1']

    else:
        model = model_dict[args.model_type].from_pretrained(args.bert_model, num_labels=num_labels)
        if args.local_rank == 0:
            torch.distributed.barrier()

        if args.fp16:
            model.half()

        model.to(device)
        model.eval()
        final_result = pred_eval('load fine-tuned', args, logger, tokenizer, model, eval_dataloader, eval_gold, label_list, device, task_name, eval_type='test')

    output_eval_file = os.path.join(args.output_dir, "Test_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Test results *****")
        for key in sorted(final_result.keys()):
            logger.info("  %s = %s", key, str(final_result[key]))
            writer.write("%s = %s\n" % (key, str(final_result[key])))

if __name__ == "__main__":
    main()
