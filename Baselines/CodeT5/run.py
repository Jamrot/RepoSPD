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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time

from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, matthews_corrcoef, roc_auc_score

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
# from evaluator import smooth_bleu
# from evaluator.CodeBLEU import calc_code_bleu
# from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist

start_time = time.time()

logging.basicConfig(filename='./test_load_data_time_.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def eval_metrics_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info(
        "  ***** Running evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    f1score, recall, precision, fpr = 0.0, 0.0, 0.0, 1.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval f1, recall, precision, fpr, auc for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(
        id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        labels = [ex.target for ex in eval_examples]
        golds = [target_dict[ex.target] for ex in eval_examples]

        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        eval_f1 = f1_score(golds, pred_nls, pos_label='true')
        eval_recall = recall_score(golds, pred_nls, pos_label='true')
        eval_precision = precision_score(golds, pred_nls, pos_label='true')
        # eval_auc = roc_auc_score(golds, pred_nls, labels=['false', 'true'])

        
        conf_matrix = confusion_matrix(golds, pred_nls, labels=['false', 'true'])
        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        eval_fpr = FP / ((FP + TN)+0.0001)

        result = {'acc':eval_acc,
                  'f1': eval_f1,
                  'recall': eval_recall,
                  'precision': eval_precision,
                  'fpr': eval_fpr,
                #   'auc': eval_auc
                  }

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)


    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

def main():
    os.environ["CUDA_LAUNCH_BLOCKING"]= "1"
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename = get_filenames(args.data_dir, args.task, args.sub_task, 'train')
    args.dev_filename = get_filenames(args.data_dir, args.task, args.sub_task, 'dev')
    args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task, 'test')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir,
                                        '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        logger.info(f'Start loading train dataset [{time.time()-start_time}]')
        train_examples, train_data = load_and_cache_gen_data(
            args, args.train_filename, pool, tokenizer, 'train')
        logger.info(f'Finish loading train dataset [{time.time()-start_time}]')

        # eval_examples, eval_data = load_and_cache_gen_data(
        #                 args, args.dev_filename, pool, tokenizer, 'dev')
        # logger.info(f'Finish loading dev dataset [{time.time()-start_time}]')
        exit(0)

        train_sampler = RandomSampler(
            train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * \
            len(train_dataloader)
        print('start warmup')
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        print('finish warmup')
        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(
            train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step = 0
        best_acc, best_f1, best_precision, best_recall, best_fpr = 0, 0, 0, 0, 1.0
        not_f1_inc_cnt = 0

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(
                train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            logger.info(f'Start training [{time.time()-start_time}s]')
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(
                        tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(
                        cur_epoch, round(train_loss, 3)))
            logger.info(f'Finish training [{time.time()-start_time}s]')
            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    logger.info(f'Start loading dev dataset [{time.time()-start_time}]')
                    eval_examples, eval_data = load_and_cache_gen_data(
                        args, args.dev_filename, pool, tokenizer, 'dev')
                    logger.info(f'Finish loading dev dataset [{time.time()-start_time}]')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                # 输出指标
                logger.info(f'Start inference [{time.time()-start_time}s]')
                result = eval_metrics_epoch(
                    args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                logger.info(f'Finish inference [{time.time()-start_time}s]')

                logger.info("  " + "*" * 20)

                if args.data_num == -1:
                    tb_writer.add_scalar('dev_acc', result['acc'], cur_epoch)
                    tb_writer.add_scalar('dev_f1', result['f1'], cur_epoch)
                    tb_writer.add_scalar('dev_recall', result['recall'], cur_epoch)
                    tb_writer.add_scalar('dev_precision', result['precision'], cur_epoch)


                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(
                        args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    output_model_file = os.path.join(
                        last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s",
                                output_model_file)

                if result['acc'] > best_acc :
                    best_acc = result['acc']
                    logger.info("  Best acc change into %s", best_f1)
                    logger.info("  " + "*" * 20)
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    acc_mdl_dir = os.path.join(args.output_dir, 'checkpoint-acc')
                    if not os.path.exists(acc_mdl_dir):
                        os.makedirs(acc_mdl_dir)
                    output_model_file = os.path.join(
                        acc_mdl_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best-acc model into %s",
                                output_model_file)

                
                if result['f1'] > best_f1 :
                    best_f1 = result['f1']
                    logger.info("  Best f1 change into %s", best_f1)
                    logger.info("  " + "*" * 20)
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    f1_mdl_dir = os.path.join(args.output_dir, 'checkpoint-f1')
                    if not os.path.exists(f1_mdl_dir):
                        os.makedirs(f1_mdl_dir)
                    output_model_file = os.path.join(
                        f1_mdl_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best-f1 model into %s",
                                output_model_file)
                    
                # if result['recall'] > best_recall :
                #     best_recall = result['recall']
                #     logger.info("  Best recall change into %s", best_recall)
                #     logger.info("  " + "*" * 20)
                    
                # if result['precision'] > best_precision :
                #     best_precision = result['precision']
                #     logger.info("  Best precision change into %s", best_precision)
                #     logger.info("  " + "*" * 20)
                    
                # if result['fpr'] < best_fpr :
                #     best_fpr = result['fpr']
                #     logger.info("  Best fpr change into %s", best_fpr)
                #     logger.info("  " + "*" * 20)
            
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
            exit(0)
        logger.info("  " + "*" * 20)
        logger.info(f'  Best acc: {best_acc}')
        logger.info(f'  Best pcs: {best_precision}')
        logger.info(f'  Best f1 : {best_f1}')
        logger.info(f'  Best fpr: {best_fpr}')
        logger.info(f'  Best rec: {best_recall}')

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['acc','f1']:
            # for pretrained_models
            file = os.path.join(
                args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            if not os.path.exists(file):
                continue
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result = eval_metrics_epoch(
                args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            print("  acc: %s", result['acc'])
            print("  pcs: %s", result['precision'])
            print("  f1 : %s", result['f1'])
            print("  fpr: %s", result['fpr'])
            print("  auc: %s", result['auc'])
            

    logger.info("Finish and take {}".format(get_elapse_time(t0)))


if __name__ == "__main__":
    main()
