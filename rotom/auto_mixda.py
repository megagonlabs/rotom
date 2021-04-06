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
from .policy import AugmentPolicyNetV4
from snippext.train_util import *
from snippext.dataset import *
from snippext.baseline import train as train_baseline
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from copy import deepcopy
from apex import amp


def sharpen(logits, T=0.5):
    logits = logits.pow(1/T)
    logits = logits / logits.sum(dim=-1, keepdim=True)
    return logits

def create_batches(l_set,
                   w_aug_set,
                   s_aug_set,
                   u_set,
                   batch_size=32):
    """Create batches for AutoMixDA

    Args:
        l_set (SnippextDataset): the labeled set
        w_aug_set (SnippextDataset): the augmented labeled set (weak, less noise)
        s_aug_set (SnippextDataset): the augmented labeled set (strong, more noise)
        u_set (SnippextDataset): the unlabeled set

    Returns:
        list of list of tuples: the mixed batches
    """
    size = len(l_set)
    datasets = [l_set, w_aug_set, s_aug_set, u_set]
    indices = [np.random.permutation(len(ds)) for ds in datasets]

    # one additional for the original unlabeled batch
    mini_batch = [[], [], [], [], []]
    padder = l_set.pad

    for i in range(size):
        for batch, dataset, index in zip(mini_batch, datasets, indices):
            batch.append(dataset[index[i]])

        # the original unlabled example
        mini_batch[-1].append(u_set.get(indices[-1][i], []))

        if i == size - 1 or len(mini_batch[0]) * 4 == batch_size:
            batch = []
            for mb in mini_batch:
                batch += mb
                mb.clear()
            yield padder(batch)


def auto_mixda(model, batch, policy=None, get_ind=False, no_ssl=False):
    """Perform one iteration of MixDA
    Args:
        model (MultiTaskNet): the model state
        batch (tuple): the input batch
        policy (AugmentPolicyNetV4, optional): the augmentation policy
    Returns:
        Tensor: the loss (of 0-d)
    """
    _, x, _, _, _, y, _, taskname = batch
    taskname = taskname[0]
    num_classes = model.num_classes

    if policy is None:
        logits, y, _ = model(x, y, task=taskname)
        if 'sts-b' not in taskname.lower():
            y = F.one_hot(y, num_classes).float()
        ind = torch.ones(x.size()[0],).to(model.device)
    else:
        # 5 batches: labeled, weakly augmented, strongly augmented,
        # unlabeled, and unlabeled original
        batch_size = x.size()[0]
        bs = batch_size // 5

        # generate model predictions
        x_enc = model(x=x[:bs*4], task=taskname, get_enc=True)
        logits, y, _ = model(x_enc=x_enc, y=y, task=taskname)
        # logits, y, _ = model(x=x[:bs*3], y=y, task=taskname)

        # convert to onehot labels for the first 3 labeled batches
        if 'sts-b' not in taskname.lower():
            y_onehot = F.one_hot(y[:bs*3], num_classes).float()
        else:
            y_onehot = y[:bs*3]

        # generate guess labels for the unlabeled batch
        with torch.no_grad():
            # u_logits, _, _ = model(x=x[bs*4:], y=y, task=taskname)
            # interpolate
            u_enc = model(x=x[bs*4:], task=taskname, get_enc=True)
            # MixDA
            beta = torch.distributions.beta.Beta(0.8, 0.8)
            lam = beta.sample()
            lam = torch.max(1 - lam, lam)
            u_enc = u_enc * lam + x_enc[3*bs:4*bs] * (1.0 - lam)

            u_logits, _, _ = model(x_enc=u_enc, y=y, task=taskname)

            if 'sts-b' not in taskname.lower():
                u_guess = sharpen(u_logits.softmax(dim=-1))
            else:
                u_guess = u_logits.view(-1)

        y = torch.cat((y_onehot, u_guess))
        # y = y_onehot

        # the policy model generate the weights of each example
        ind = policy(x[:bs*4], y, x_enc=x_enc[:bs*4], prediction=logits.softmax(dim=-1))
        if no_ssl:
            ind[-bs:] = torch.zeros(bs,).to(model.device)

    # consider three types of tasks: tagging, regression, and classification
    if 'sts-b' in taskname.lower():
        logits = logits.view(-1)
        loss = (((logits - y) ** 2).sum() * ind).mean() # F.mse_loss(logits, y)
    else:
        logits = logits.view(-1, logits.shape[-1])
        logits = F.softmax(logits, dim=-1)
        if 'tagging' in taskname:
            sz = (y.size()[2] - 1) * y.size()[1]
            ind = ind.view(-1).repeat_interleave(sz).view(-1, y.size()[2] - 1)
            y = y.view(-1, y.shape[-1])
            loss = torch.sum(-y[:, 1:] * logits[:, 1:].log() * ind, dim=-1).mean() # TODO: this one is problematic
        else:
            loss = (torch.sum(-y * logits.log(), -1) * ind).mean()

    if get_ind:
        return loss, ind.detach()
    else:
        return loss


def train(model,
          policy,
          l_set,
          w_aug_set,
          s_aug_set,
          u_set,
          v_set,
          optimizer,
          policy_optimizer,
          scheduler=None,
          policy_scheduler=None,
          fp16=False,
          batch_size=32,
          no_ssl=False):
    """Perform one epoch of AutoMixDA

    Args:
        model (MultiTaskModel): the model state
        policy (AugmentPolicyNetV4): the policy model
        l_set (SnippextDataset): the labeled set
        w_aug_set (SnippextDataset): the augmented labeled set (weak, less noise)
        s_aug_set (SnippextDataset): the augmented labeled set (strong, more noise)
        u_set (SnippextDataset): the unlabeled set
        v_set (SnippextDataset): the validation set
        optimizer (Optimizer): Adam
        policy_optimizer (Optimizer): Adam
        scheduler (Optional):
        policy_scheduler (Optional):
        fp16 (boolean, Optional): whether to use fp16
        batch_size (int, Optional): batch size
        no_ssl (bool, Optional): if True, then don't use unlabeled data

    Returns:
        None
    """
    da_batches = None
    # create the training batches
    batches = create_batches(l_set,
                             w_aug_set,
                             s_aug_set,
                             u_set,
                             batch_size)

    model.train()
    i = 0
    ind_stat = np.zeros(4)

    for batch in batches:
        try:
            da_batch = next(da_batches)
        except:
            da_batches = iter(data.DataLoader(dataset=v_set,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=v_set.pad))
            da_batch = next(da_batches)

        words, x, is_heads, tags, mask, y, seqlens, taskname = batch
        taskname = taskname[0]
        _y = y

        # phase one: update the policy parameters

        # make a copy of the current model values
        model_values = [p.data.clone() for p in model.parameters()]

        # learning rate
        lr = [group['lr'] for group in optimizer.param_groups]

        # virtual SGD step
        optimizer.zero_grad()
        train_loss = auto_mixda(model, batch, policy=policy, no_ssl=no_ssl)
        if fp16:
            with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            train_loss.backward()

        # optimizer.step()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data = p.data - lr[0] * p.grad.data

        # validation loss
        optimizer.zero_grad()
        val_loss = auto_mixda(model, da_batch, policy=None)
        if fp16:
            with amp.scale_loss(val_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            val_loss.backward()

        # compute epsilon
        model_gradients = [p.grad.data.clone() if p.grad is not None \
                else None for p in model.parameters()]
        with torch.no_grad():
            grad_L2 = torch.zeros(())
            for g in model_gradients:
                if g is not None:
                    grad_L2 += g.pow(2).sum()

            epsilon = 0.01 / grad_L2.sqrt()

        # w_plus and w_minus
        policy_optimizer.zero_grad()
        for sign in [-1.0, 1.0]:
            with torch.no_grad():
                for p, g, v in zip(model.parameters(), model_gradients, model_values):
                    if g is not None:
                        p.data = v + sign * epsilon * g
                    else:
                        p.data = v
            loss_pm = auto_mixda(model, batch, policy=policy, no_ssl=no_ssl) # using model's batch
            loss_pm = loss_pm * -sign / 2 / epsilon * lr[0]
            if fp16:
                with amp.scale_loss(loss_pm, policy_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_pm.backward()

        # optimize
        policy_optimizer.step()
        # if policy_scheduler:
        #     policy_scheduler.step()

        # reset the model's parameters
        with torch.no_grad():
            for p, v in zip(model.parameters(), model_values):
                p.data = v

        # phase two: update the model parameters
        optimizer.zero_grad()
        loss, ind = auto_mixda(model, batch, policy=policy, no_ssl=no_ssl, get_ind=True)
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # debug: v_loss should decrease
        # with torch.no_grad():
        #     v_loss_after = auto_mixda(model, da_batch, policy=None)
        # print("v_loss before: ", val_loss.detach().cpu().numpy(), "v_loss after: ", v_loss_after.detach().cpu().numpy())

        # # get ind stat
        # ind_stat += ind.view(4,-1).sum(dim=-1).cpu().numpy()
        # print(ind_stat)

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
            max_idx = int(ind.argmax().cpu())
            min_idx = int(ind.argmin().cpu())
            print(ind, ind.max(), ind.min())
            bs = len(words) // 4
            ds_names = ['original', 'w_aug', 's_aug', 'unlabeled']
            if 'tagging_' not in taskname:
                print('good (%s), y=%d: %s' % (ds_names[max_idx // bs], int(_y.cpu()[max_idx]), words[max_idx]))
                print('bad (%s), y=%d: %s' % (ds_names[min_idx // bs], int(_y.cpu()[min_idx]), words[min_idx]))

            print("=======================")

        if i%10 == 0: # monitoring
            print(f"step: {i}, task: {taskname}, loss: {loss.item()}")
            del loss
        i += 1


def initialize_and_train(task_config,
                         l_set,
                         w_aug_set,
                         s_aug_set,
                         u_set,
                         validset,
                         testset,
                         hp,
                         run_tag):
    """The train process.

    Args:
        task_config (dictionary): the configuration of the task
        l_set (SnippextDataset): the labeled set
        w_aug_set (SnippextDataset): the augmented labeled set (weak, less noise)
        s_aug_set (SnippextDataset): the augmented labeled set (strong, more noise)
        u_set (SnippextDataset): the unlabeled set
        validset (SnippextDataset): the validation set
        testset (SnippextDataset): the testset
        hp (Namespace): the parsed hyperparameters
        run_tag (string): the tag of the run (for logging purpose)

    Returns:
        None
    """
    # iterators for dev/test set
    padder = SnippextDataset.pad
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*4,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model
    if l_set.vocab is None:
        num_classes = 0
    else:
        num_classes = len(l_set.vocab)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MultiTaskNet([task_config], device,
                         lm=hp.lm, bert_path=hp.bert_path)
    policy = AugmentPolicyNetV4(num_classes, device,
                         lm=hp.lm, bert_path=hp.bert_path)

    # move to device
    model = model.to(device)
    policy = policy.to(device)

    model_optimizer = AdamW(model.parameters(), lr=hp.lr)
    policy_optimizer = AdamW(policy.parameters(), lr=hp.lr)
    if device == 'cuda' and hp.fp16:
        model, model_optimizer = amp.initialize(model,
                                          model_optimizer,
                                          opt_level='O2')
        policy, policy_optimizer = amp.initialize(policy,
                                          policy_optimizer,
                                          opt_level='O2')

    # create logging
    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    writer = SummaryWriter(log_dir=hp.logdir)

    # start training
    best_dev_f1 = best_test_f1 = 0.0
    epoch = 1

    # warmup: no DA or SSL for the first half of epochs
    if hp.warmup:
        # learning rate scheduler
        num_steps = (len(l_set) // hp.batch_size) * hp.n_epochs // 2
        scheduler = get_linear_schedule_with_warmup(model_optimizer,
                                                    num_warmup_steps=num_steps // 10,
                                                    num_training_steps=num_steps)

        while epoch <= hp.n_epochs // 2:
            train_baseline(model,
                  l_set,
                  model_optimizer,
                  scheduler=scheduler,
                  batch_size=hp.batch_size,
                  fp16=hp.fp16)

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

            # if dev_f1 > 1e-6:
            epoch += 1
            if hp.save_model:
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    torch.save(model.state_dict(), run_tag + '_dev.pt')
                if test_f1 > best_test_f1:
                    best_test_f1 = dev_f1
                    torch.save(model.state_dict(), run_tag + '_test.pt')
    # Testing
    # policy.bert = model.bert

    # learning rate scheduler
    if hp.warmup:
        num_steps = (4 * len(l_set) // hp.batch_size) * hp.n_epochs // 2
    else:
        num_steps = (4 * len(l_set) // hp.batch_size) * hp.n_epochs

    model_scheduler = get_linear_schedule_with_warmup(model_optimizer,
                                      num_warmup_steps=0,
                                      num_training_steps=num_steps)
    policy_scheduler = get_linear_schedule_with_warmup(policy_optimizer,
                                      num_warmup_steps=0,
                                      num_training_steps=num_steps)


    # best_dev_f1 = best_test_f1 = 0.0
    # train the model and policy
    while epoch <= hp.n_epochs:
        train(model,
              policy,
              l_set,
              w_aug_set,
              s_aug_set,
              u_set,
              validset,
              model_optimizer,
              policy_optimizer,
              scheduler=model_scheduler,
              policy_scheduler=policy_scheduler,
              fp16=hp.fp16,
              batch_size=hp.batch_size,
              no_ssl=(hp.no_ssl or 'no_ssl' in hp.da))

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
        epoch += 1

    writer.close()
