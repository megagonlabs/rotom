import os
import argparse
import json
import sys
import numpy as np
import random
import torch

from rotom.dataset import TextCLSDataset
from ditto.dataset import DittoDataset
from functools import partial

num_classes = {'AMAZON2': 2, 'AMAZON5': 5,
               'AG': 4}

vocabs = {'SNIPS': ['AddToPlaylist', 'BookRestaurant',
                 'GetWeather', 'PlayMusic',
                 'RateBook', 'SearchCreativeWork',
                 'SearchScreeningEvent'],
          'ATIS': ['atis_abbreviation', 'atis_aircraft',
              'atis_airfare', 'atis_airline',
              'atis_airline#atis_flight_no', 'atis_airport',
              'atis_capacity', 'atis_city', 'atis_distance',
              'atis_flight', 'atis_flight#atis_airfare',
              'atis_flight_no', 'atis_flight_time',
              'atis_ground_fare', 'atis_ground_service',
              'atis_quantity', 'atis_restriction',
              'atis_meal',
              'atis_day_name',
              'atis_airfare#atis_flight',
              'atis_flight#atis_airline',
              'atis_flight_no#atis_airline',
              'atis_airfare#atis_flight_time',
              'atis_ground_service#atis_ground_fare'],
          'TREC': ['0', '1', '2', '3', '4', '5'],
          'SST-2': ['0', '1'],
          'SST-5': ['0', '1', '2', '3', '4'],
          'IMDB': ['pos', 'neg']}


def get_cls_config(hp):
    """Get configuration of the task"""
    taskname = hp.task
    if 'em_' in taskname:
        name = taskname[3:]
        vocab = ['0', '1']
        path = 'data/em/%s/' % name
        config = {'name': taskname,
                'trainset': path + 'train.txt',
                'validset': path + 'valid.txt',
                'testset': path + 'test.txt',
                'task_type': 'classification',
                'vocab': vocab}

        config['unlabeled'] = config['trainset']
        config['validset'] = config['trainset']
        return config,\
               DittoDataset,\
               DittoDataset
    elif 'cleaning_' in taskname:
        LL = taskname.split('_')
        if hp.size is not None:
            size, idx = str(hp.size), str(hp.run_id)
            name = LL[1]
        else:
            prefix, size, idx = LL[0], LL[-2], LL[-1]
            name = '_'.join(LL[1:-2])

        path = 'data/cleaning/%s/%s_10000/%s/' % (name, size, idx)
        vocab = ['0', '1']
        config = {'name': taskname,
                  'trainset': path + 'train.txt',
                  'validset': path + 'train.txt',
                  'testset': path + 'test.txt',
                  'unlabeled': path + 'unlabeled.txt',
                  'task_type': 'classification',
                  'vocab': vocab}
        return config, DittoDataset, DittoDataset
    elif 'compare' in taskname:
        # compare2_SST-2
        LL = taskname.split('_')
        prefix, name = LL[0], LL[1]
        path = 'data/textcls/%s/%s/' % (prefix, name)
        vocab = vocabs[name]
        idx = str(hp.run_id)
        config = {'name': taskname,
                  'trainset': path + 'train.txt.%s' % idx,
                  'validset': path + 'valid.txt.%s'  % idx,
                  'testset': path + 'test.txt',
                  'unlabeled': path + 'train.txt.full',
                  'task_type': 'classification',
                  'vocab': vocab}
        return config, TextCLSDataset, TextCLSDataset
    else:
        # Text CLS datasets
        if 'textcls_' in taskname:
            taskname = taskname.replace('textcls_', '')
        if hp.size is None:
            path, size = taskname.split('_')
        else:
            path = taskname
            size = str(hp.size)
        path = path.upper()
        if path in vocabs:
            vocab = vocabs[path]
        else:
            vocab = [str(i) for i in \
                    range(1, num_classes[path]+1)]
        path = 'data/textcls/%s' % path

        config = {'name': taskname,
                  'trainset': '%s/train.txt.%s' % (path, size),
                  'validset': '%s/valid.txt.%s' % (path, size),
                  'testset': '%s/test.txt' % path,
                  'unlabeled': '%s/train.txt.full' % path,
                  'task_type': 'classification',
                  'vocab': vocab}
        return config, TextCLSDataset, TextCLSDataset

def get_ops(hp):
    """return a pair of DA operators for each task"""
    em = ["t5", "del", "del"]
    cls = ["t5", "token_repl_tfidf", "token_del_tfidf"]
    cleaning = ["t5", "swap", "swap"]

    if 'cleaning_' in task:
        return cleaning
    if "em_" in task: # EM
        return em
    else:
        return cls

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="mrpc")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    # for ssl
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--num_aug", type=int, default=2)
    parser.add_argument("--u_lambda", type=float, default=10.0)
    # for no ssl ablation
    parser.add_argument("--no_ssl", dest="no_ssl", action="store_true")
    # for dataset balancing
    parser.add_argument("--balance", dest="balance", action="store_true")
    # warmup
    parser.add_argument("--warmup", dest="warmup", action="store_true")

    hp = parser.parse_args()

    # set seed
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # create the tag of the run
    if hp.no_ssl:
        run_tag = '%s_lm=%s_da=%s_no_ssl_alpha=%.1f_id=%d' % (hp.task, hp.lm, hp.da, hp.alpha_aug, hp.run_id)
    else:
        run_tag = '%s_lm=%s_da=%s_alpha=%.1f_id=%d' % (hp.task, hp.lm, hp.da, hp.alpha_aug, hp.run_id)
    if hp.size is not None:
        run_tag += '_size=%d' % hp.size

    config, Dataset, TestDataset = get_cls_config(hp)

    if hp.balance:
        Dataset = partial(Dataset, balance=hp.balance)

    task = config['name']
    vocab = config['vocab']
    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    task_type = config['task_type']

    if hp.da == 'edbt20':
        trainset += '.no_header'
        validset += '.no_header'
        testset += '.no_header'

    train_dataset = Dataset(trainset, vocab, task,
            lm=hp.lm, max_len=hp.max_len, size=hp.size)

    # valid_dataset = Dataset(validset, vocab, task,
    valid_dataset = TestDataset(validset, vocab, task,
                                    lm=hp.lm, max_len=hp.max_len, size=hp.size)
    test_dataset = TestDataset(testset, vocab, task,
                                   lm=hp.lm, max_len=hp.max_len)

    # get default DA's
    ops = get_ops(hp)

    if hp.da is None or hp.da in ['None', 'edbt20']:
        # No DA
        from snippext.baseline import initialize_and_train
        initialize_and_train(config,
                             train_dataset,
                             valid_dataset,
                             test_dataset,
                             hp,
                             run_tag)
    elif 'auto_ssl' in hp.da or 'auto_filter_weight' in hp.da:
        if 'em_' in task or 'compare' in task:
            # a lightweight version for faster EM experiments
            from rotom.auto_mixda import initialize_and_train
        else:
            from rotom.auto_filter_weight import initialize_and_train

        # the augmented training set
        w_aug_set = Dataset(trainset, vocab, task,
                size=hp.size,
                lm=hp.lm, max_len=hp.max_len, augment_op=ops[0])

        s_aug_set = Dataset(trainset, vocab, task,
                size=hp.size,
                lm=hp.lm, max_len=hp.max_len, augment_op=ops[1])

        # unlabeled dataset and augmented
        unlabeled = config['unlabeled']
        u_set = Dataset(unlabeled, vocab, task,
                                   max_len=hp.max_len,
                                   lm=hp.lm,
                                   augment_op=ops[2],
                                   size=10000)

        # train the model
        initialize_and_train(config,
                             train_dataset,
                             w_aug_set,
                             s_aug_set,
                             u_set,
                             valid_dataset,
                             test_dataset,
                             hp, run_tag)
    else: # normal DA or InvDA
        augment_dataset = Dataset(trainset, vocab, task,
                lm=hp.lm, max_len=hp.max_len,
                augment_op=hp.da, size=hp.size)

        if abs(hp.alpha_aug) < 1e-6:
            # no DA
            from snippext.baseline import initialize_and_train
            initialize_and_train(config,
                                 augment_dataset,
                                 valid_dataset,
                                 test_dataset,
                                 hp,
                                 run_tag)
        else:
            # MixDA or InvDA
            from snippext.mixda import initialize_and_train
            initialize_and_train(config,
                                 train_dataset,
                                 augment_dataset,
                                 valid_dataset,
                                 test_dataset,
                                 hp,
                                 run_tag)
