import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import argparse
import json
import copy
import random

from torch.utils import data
from .model import MultiTaskNet
from snippext.train_util import *
from snippext.dataset import *
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from apex import amp

# criterion for tagging
tagging_criterion = nn.CrossEntropyLoss(ignore_index=0)

# criterion for classification
classifier_criterion = nn.CrossEntropyLoss()

# criterion for regression
regression_criterion = nn.MSELoss()

def mixda(model, batch, alphas=[]):
    """Perform one iteration of MixDA

    Args:
        model (MultiTaskNet): the model state
        batch (tuple): the input batch
        alphas (list of float, Optional): the parameters for MixDA

    Returns:
        Tensor: the loss (of 0-d)
    """
    _, x, _, _, mask, y, _, taskname = batch
    taskname = taskname[0]

    # number of operators
    num_ops = len(alphas) + 1

    # two batches
    batch_size = x.size()[0] // num_ops

    # get the BERT encoding
    x_enc = model(x=x, task=taskname, get_enc=True)
    # x_enc = model(x=x, task=taskname, get_emb=True)

    x_enc_now = x_enc[:batch_size]
    y = y[:batch_size]

    for i, alpha in enumerate(alphas):
        x_enc_next = x_enc[batch_size * (i + 1): batch_size * (i + 2)]
        # sample alpha
        aug_lam = np.random.beta(alpha, alpha)
        # interpolate
        x_enc_now = x_enc_now * aug_lam + x_enc_next * (1.0 - aug_lam)

    # forward
    logits, y, _ = model(y=y,
                         x_enc=x_enc_now,
                         # x_emb=x_enc_now,
                         task=taskname)
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)

    # consider three types of tasks: tagging, regression, and classification
    if 'tagging' in taskname:
        criterion = tagging_criterion
    elif 'sts-b' in taskname:
        criterion = regression_criterion
    else:
        criterion = classifier_criterion

    loss = criterion(logits, y)
    return loss


def create_mixda_batches(l_set, policy, batch_size=16):
    """Create batches for mixda

    Each batch is the concatenation of (1) a labeled batch and (2) an augmented
    labeled batch (having the same order of (1) )

    Args:
        l_set (SnippextDataset): the train set
        policy (SnippextDataset): the augmentation policy
        batch_size (int, optional): batch size (of each component)

    Returns:
        list of list: the created batches
    """
    mixed_batches = []
    num_labeled = len(l_set)
    l_index = np.random.permutation(num_labeled)
    num_ops = len(policy['ops'])

    sub_batches = [[] for _ in range(num_ops)]
    padder = l_set.pad

    for i, idx in enumerate(l_index):
        for j, op in enumerate(policy['ops']):
            sub_batches[j].append(l_set.get(idx, op))

        if len(sub_batches[0]) == batch_size or i == len(l_index) - 1:
            batch = []
            for sb in sub_batches:
                batch += sb
            mixed_batches.append(padder(batch))
            for sb in sub_batches:
                sb.clear()

    random.shuffle(mixed_batches)
    return mixed_batches


def train(model, l_set, policy, optimizer,
          scheduler=None,
          fp16=False,
          batch_size=32):
    """Perform one epoch of MixDA

    Args:
        model (MultiTaskModel): the model state
        train_dataset (SnippextDataset): the train set
        policy (Dict): the augmentation policy
        optimizer (Optimizer): Adam
        fp16 (boolean, Optional): whether to use fp16
        batch_size (int, Optional): batch size

    Returns:
        None
    """
    mixda_batches = create_mixda_batches(l_set,
                                         policy,
                                         batch_size=batch_size)
    alphas = policy['alphas']

    model.train()
    for i, batch in enumerate(mixda_batches):
        # for monitoring
        words, x, is_heads, tags, mask, y, seqlens, taskname = batch
        taskname = taskname[0]
        _y = y

        # perform mixmatch
        optimizer.zero_grad()
        loss = mixda(model, batch, alphas)
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if i == 0:
            print("=====sanity check======")
            print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", get_tokenizer().convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            y_sample = _y.cpu().numpy()[0]
            if np.isscalar(y_sample):
                print("y:", y_sample)
            else:
                print("y:", y_sample[:seqlens[0]])
            print("tags:", tags[0])
            print("mask:", mask[0])
            print("seqlen:", seqlens[0])
            print("task_name:", taskname)
            print("=======================")

        if i%10 == 0: # monitoring
            print(f"step: {i}, task: {taskname}, loss: {loss.item()}")
            del loss



def initialize_and_train(task_config,
                         trainset,
                         policy,
                         validset,
                         testset,
                         hp,
                         run_tag):
    """The train process.

    Args:
        task_config (dictionary): the configuration of the task
        trainset (SnippextDataset): the training set
        policy (Dict): the data augmentation policy
        validset (SnippextDataset): the validation set
        testset (SnippextDataset): the testset
        hp (Namespace): the parsed hyperparameters
        run_tag (string): the tag of the run (for logging purpose)

    Returns:
        None
    """
    padder = SnippextDataset.pad

    # iterators for dev/test set
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)


    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        model = MultiTaskNet([task_config], device,
                         lm=hp.lm, bert_path=hp.bert_path)
        optimizer = AdamW(model.parameters(), lr=hp.lr)
    else:
        model = MultiTaskNet([task_config], device,
                         lm=hp.lm, bert_path=hp.bert_path).cuda()
        optimizer = AdamW(model.parameters(), lr=hp.lr)
        if hp.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # learning rate scheduler
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)
    # create logging
    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    writer = SummaryWriter(log_dir=hp.logdir)

    # start training
    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        train(model,
              trainset,
              policy,
              optimizer,
              scheduler=scheduler,
              fp16=hp.fp16,
              batch_size=hp.batch_size)

        print(f"=========eval at epoch={epoch}=========")
        dev_f1, test_f1 = eval_on_task(epoch,
                            model,
                            task_config['name'],
                            valid_iter,
                            validset,
                            test_iter,
                            testset,
                            writer,
                            run_tag)

        if hp.save_model:
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                torch.save(model.state_dict(), run_tag + '_dev.pt')
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                torch.save(model.state_dict(), run_tag + '_test.pt')

    writer.close()

