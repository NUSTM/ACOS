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
from dataset_utils import *

gm = GPUManager()
device = gm.auto_choice(mode=0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss, BCEWithLogitsLoss

from modeling import CategorySentiClassification

# sys.path.insert(0, '/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT')
# from modeling_for_share import BertForQuadABSAPairCSAO
from bert_utils.tokenization import BertTokenizer
from bert_utils.optimization import BertAdam, WarmupLinearSchedule

from run_classifier_dataset_utils import *
from eval_metrics import *
import gc

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import warnings

warnings.filterwarnings('ignore')

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
    parser.add_argument("--bert_model", default=None, type=str, required=True)
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
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
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        # default=42,
                        default=13,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    args = parser.parse_args()
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    label_list = processor.get_labels(args.domain_type)
    num_labels = len(label_list[0])

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model_dict = {
        'categorysenti': CategorySentiClassification,
    }
    cate_dict = {label : i for i, label in enumerate(label_list[0])}

    global_step = 0
    nb_tr_steps = 0
    eval_quad_gold = []
    train_quad_gold = []
    eval_quad_text = []
    train_quad_text = []
    #for entity#attribute
    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir, args.domain_type)
        f = cs.open(args.data_dir+'/tokenized_data/'+args.domain_type+'_test_pair.tsv', 'r').readlines()
        eval_quad_text, eval_quad_gold = read_pair_gold(f, args)

        eval_features = convert_examples_to_features2nd(
            eval_examples, label_list, args.max_seq_length, tokenizer, task_name)

        eval_tokens_len = torch.tensor([f.tokens_len for f in eval_features], dtype=torch.long)
        eval_aspect_input_ids = torch.tensor([f.aspect_input_ids for f in eval_features], dtype=torch.long)
        eval_aspect_input_mask = torch.tensor([f.aspect_input_mask for f in eval_features], dtype=torch.long)
        eval_aspect_segment_ids = torch.tensor([f.aspect_segment_ids for f in eval_features], dtype=torch.long)
        eval_candidate_aspect = torch.tensor([f.candidate_aspect for f in eval_features], dtype=torch.long)
        eval_candidate_opinion = torch.tensor([f.candidate_opinion for f in eval_features], dtype=torch.long)
        eval_label_id = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        # eval_gold = [eval_aspect_input_ids.numpy().tolist(), eval_quad_gold]
        eval_gold = [eval_quad_text, eval_quad_gold]

        eval_data = TensorDataset(eval_tokens_len, eval_aspect_input_ids, eval_aspect_input_mask,
        eval_aspect_segment_ids, eval_candidate_aspect, eval_candidate_opinion,
        eval_label_id)
        # Run prediction for full data
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)  # Note that this sampler samples randomly
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    
    # Prepare data loader
    train_examples = processor.get_train_examples(args.data_dir, args.domain_type)
    train_features = convert_examples_to_features2nd(
        train_examples, label_list, args.max_seq_length, tokenizer, task_name)

    tokens_len = torch.tensor([f.tokens_len for f in train_features], dtype=torch.long)
    aspect_input_ids = torch.tensor([f.aspect_input_ids for f in train_features], dtype=torch.long)
    aspect_input_mask = torch.tensor([f.aspect_input_mask for f in train_features], dtype=torch.long)
    aspect_segment_ids = torch.tensor([f.aspect_segment_ids for f in train_features], dtype=torch.long)
    candidate_aspect = torch.tensor([f.candidate_aspect for f in train_features], dtype=torch.long)
    candidate_opinion = torch.tensor([f.candidate_opinion for f in train_features], dtype=torch.long)
    label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    valid_examples = processor.get_valid_examples(args.data_dir, args.domain_type)
    valid_features = convert_examples_to_features2nd(
        valid_examples, label_list, args.max_seq_length, tokenizer, task_name)
    f = cs.open(args.data_dir+'/tokenized_data/'+args.domain_type+'_dev_pair.tsv', 'r').readlines()
    valid_quad_text, valid_quad_gold = read_pair_gold(f, args)

    valid_tokens_len = torch.tensor([f.tokens_len for f in valid_features], dtype=torch.long)
    valid_aspect_input_ids = torch.tensor([f.aspect_input_ids for f in valid_features], dtype=torch.long)
    valid_aspect_input_mask = torch.tensor([f.aspect_input_mask for f in valid_features], dtype=torch.long)
    valid_aspect_segment_ids = torch.tensor([f.aspect_segment_ids for f in valid_features], dtype=torch.long)
    valid_candidate_aspect = torch.tensor([f.candidate_aspect for f in valid_features], dtype=torch.long)
    valid_candidate_opinion = torch.tensor([f.candidate_opinion for f in valid_features], dtype=torch.long)
    valid_label_id = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)

    all_results = []


    valid_gold = [valid_quad_text, valid_quad_gold]

    train_data = TensorDataset(tokens_len, aspect_input_ids, aspect_input_mask,
    aspect_segment_ids, candidate_aspect, candidate_opinion, 
    label_id)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_data = TensorDataset(valid_tokens_len, valid_aspect_input_ids, valid_aspect_input_mask,
    valid_aspect_segment_ids, valid_candidate_aspect, valid_candidate_opinion, 
    valid_label_id)
    
    if args.local_rank == -1:
        valid_sampler = SequentialSampler(valid_data)
    else:
        valid_sampler = DistributedSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.eval_batch_size)

    if args.do_train:
        logger.info("***** Running training *****")

        num_train_optimization_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.num_train_epochs

        model = model_dict[args.model_type].from_pretrained(args.bert_model, num_labels=num_labels)
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

        if args.local_rank == 0:
            torch.distributed.barrier()

        model.to(device)
        max_macro_F1 = -1.0

        for _e in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                _tokens_len, _aspect_input_ids, _aspect_input_mask, _aspect_segment_ids, _candidate_aspect, \
                _candidate_opinion, _label_id = batch

                # define a new function to compute loss values for both output_modes

                losses, logits = model(tokenizer, _e, aspect_input_ids=_aspect_input_ids,
                aspect_token_type_ids=_aspect_segment_ids, aspect_attention_mask=_aspect_input_mask,
                candidate_aspect=_candidate_aspect, candidate_opinion=_candidate_opinion, label_id=_label_id)

                if step % 10 == 0:
                    logger.info('Total Loss is {} .'.format(losses[0]))
                step += 1
                loss = losses[0]
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                else:
                    loss = loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                nb_tr_examples += _aspect_input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            model.eval()
            result = pair_eval(_e, args, logger, tokenizer, model, valid_dataloader, valid_gold, 
            label_list, device, task_name, eval_type='valid')

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

                final_result = pair_eval(_e, args, logger, tokenizer, model, eval_dataloader, eval_gold, 
                label_list, device, task_name, eval_type='test')
                max_macro_F1 = result['micro-F1']
    else:
        model = model_dict[args.model_type].from_pretrained(args.bert_model, num_labels=num_labels)
        if args.local_rank == 0:
            torch.distributed.barrier()

        model.to(device)
        model.eval()
        final_result = pair_eval('load fine-tuned', args, logger, tokenizer, model, eval_dataloader, eval_gold, 
        label_list, device, task_name, eval_type='test')

    output_eval_file = os.path.join(args.output_dir, "Test_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Test results *****")
        for key in sorted(final_result.keys()):
            logger.info("  %s = %s", key, str(final_result[key]))
            writer.write("%s = %s\n" % (key, str(final_result[key])))

if __name__ == "__main__":
    main()
