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
from .policy import FilterPolicyNet, AugmentPolicyNetV4
from snippext.train_util import *
from snippext.dataset import *
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from copy import deepcopy
from apex import amp
from tqdm import tqdm


def sharpen(logits, T=0.5):
    """Sharpen a label probability distribution (make closer to onehot)

    Args:
        logits (Tensor): the input probability

    Returns:
        Tensor: the sharpened tensor
    """
    logits = logits.pow(1/T)
    logits = logits / logits.sum(dim=-1, keepdim=True)
    return logits

def sharpen_onehot(logits, bar=0.1):
    """Sharpen a label probability distribution to onehot given a threshold

    Args:
        logits (Tensor): the input probability

    Returns:
        Tensor: the sharpened tensor
    """
    T = 1.0 / logits.size()[-1] + bar

    # above bar
    res_above = logits.softmax(dim=-1)
    res_above.ge_(T)

    # below bar
    res_below = logits

    max_val = logits.max(dim=-1)[0]

    res = res_above * max_val.ge(T).float().unsqueeze(1) + \
          res_below * max_val.le(T).float().unsqueeze(1)
    return res

def create_batches(l_set,
                   w_aug_set,
                   s_aug_set,
                   u_set,
                   batch_size=32,
                   max_aug=16,
                   no_ssl=False):
    """Create batches for filtering and weighting

    Args:
        l_set (SnippextDataset): the labeled set
        w_aug_set (SnippextDataset): the augmented labeled set (weak, less noise)
        s_aug_set (SnippextDataset): the augmented labeled set (strong, more noise)
        u_set (SnippextDataset): the unlabeled set
        batch_size (int, optional): the batch size
        max_aug (int, optional): the maximal number of augmentations per example
        no_ssl (bool, optional): if true then turn off SSL

    Returns:
        list of list of tuples: the mixed batches
    """
    size = len(l_set)
    l_index = np.random.permutation(size)

    u_size = len(u_set)
    u_index = np.random.permutation(u_size)

    # half labeled and half unlabeled
    examples = []
    u_pos = 0

    # prepare the datasets
    # generate enough augmented examples
    for ds in [w_aug_set, s_aug_set]:
        if not hasattr(ds, "aug_examples"):
            ds.aug_examples = [[] for _ in range(len(ds))]

            # TODO: this part is very ugly; needs to be changed
            if hasattr(ds, "t5_examples"):
                for i in range(len(ds)):
                    for idx in range(len(ds.t5_examples[i])):
                        ds.aug_examples[i].append(ds.get(i, ops=[['t5']],
                            aug_idx=idx))
            else:
                for i in tqdm(range(len(ds))):
                    for _ in range(max_aug):
                        ds.aug_examples[i].append(ds[i])

    # build the batches
    for i in range(size):
        # add to l_batch
        idx = l_index[i]
        exm = {'original': l_set[idx]}

        # augmentations
        augments = []
        num_aug = min(max_aug // 2, len(s_aug_set.aug_examples[idx]))
        augments += random.sample(s_aug_set.aug_examples[idx], num_aug)
        num_aug = min(max_aug - num_aug, len(w_aug_set.aug_examples[idx]))
        augments += random.sample(w_aug_set.aug_examples[idx], num_aug)
        exm['augments'] = augments

        # unlabeled examples
        u_exms = []
        u_exms_aug = []

        if not no_ssl:
            while len(u_exms) < 1:
            # while len(u_exms) < max_aug:
                idx = u_index[u_pos]
                # not augmented
                u_exms.append(u_set.get(idx, []))
                # augmented
                u_exms_aug.append(u_set[idx])
                u_pos += 1
                if u_pos >= len(u_set):
                    u_pos = 0
                    u_index = np.random.permutation(u_size)

        exm['u_exms'] = u_exms
        exm['u_exms_aug'] = u_exms_aug

        # append and return
        examples.append(exm)

        # 1 labeled + 1 unlabeled
        if i == size - 1 or len(examples) == batch_size // 2:
            yield examples
            examples.clear()


def process_filter(model, policy, batch, padder):
    """Perform the filtering process

    Args:
        model (MultiTaskNet): the model state
        filter_model (FilterPolicyNet): the filtering policy
        batch (List): the list of raw batchs to be filtered
        padder: the padding function to form batches

    Returns:
        List of Tuple: the filtered batch
        Tensor: the input encodings for MixDA
        Tensor: the original or guessed labels
        Tensor: the log probability of the batch (0-d)
    """
    # step 1: predict with the model
    all_exms = []
    u_indices = []
    for exm in batch:
        all_exms.append(exm['original'])
        for aug in exm['augments']:
            all_exms.append(aug)

        for u_exm_aug in exm['u_exms_aug']:
            u_indices.append(len(all_exms))
            all_exms.append(u_exm_aug)

        for u_exm in exm['u_exms']:
            # mark the positions of unlabeled examples
            u_indices.append(len(all_exms))
            all_exms.append(u_exm)

    x_encs = []
    y_preds = []
    ys = []

    batch_size = 256
    num_classes = model.num_classes
    with torch.no_grad():
        # form the batches
        for start in range(0, len(all_exms), batch_size):
            end = min(start + batch_size, len(all_exms))
            sb = padder(all_exms[start:end])
            _, x, _, _, _, y, _, taskname = sb
            taskname = taskname[0]

            x_enc = model(x, task=taskname, get_enc=True)
            logits, y, _ = model(x_enc=x_enc, y=y, task=taskname)

            if 'sts-b' not in taskname.lower():
                y_pred = logits.softmax(dim=-1)
                y = F.one_hot(y, num_classes).float()
            else:
                y_pred = logits.view(-1)

            x_encs.append(x_enc)
            y_preds.append(y_pred)
            ys.append(y)

        # concatenate
        x_encs = torch.cat(tuple(x_encs))
        y_preds = torch.cat(tuple(y_preds))
        ys = torch.cat(tuple(ys))

        # sharpening for unlabeled y's
        if len(u_indices) > 0:
            if 'sts-b' not in taskname.lower():
                # ys[u_indices] = sharpen(y_preds[u_indices])
                # ys[u_indices] = sharpen_onehot(y_preds[u_indices])
                if random.randint(0, 1) == 0:
                    ys[u_indices] = sharpen_onehot(y_preds[u_indices])
                else:
                    ys[u_indices] = sharpen(y_preds[u_indices])
            else:
                ys[u_indices] = y_preds[u_indices]


    # step 2: featurize each (original, augmented, and unlabeled) example
    # step 3: predict with the filtering policy
    idx = 0
    new_batch = []
    exm_indices = []
    aug_mp = {}
    log_prob_sum = torch.Tensor([0.0]).to(model.device)

    for exm in batch:
        y_indices = []

        # original
        ori_idx = idx
        y_indices.append((idx, idx))
        idx += 1

        for aug in exm['augments']:
            y_indices.append((idx, ori_idx))
            idx += 1

        u_size = len(exm['u_exms_aug'])
        for u_exm in exm['u_exms_aug']:
            y_indices.append((idx, idx + u_size))
            idx += 1

        for k, v in y_indices:
            aug_mp[k] = v

        # skip the u_aug
        idx += len(exm['u_exms'])

        idx1, idx2 = zip(*y_indices)
        y = ys[list(idx2)]
        y_pred = y_preds[list(idx1)]
        y_aug = y_preds[list(idx2)]

        # featurize
        features = policy.featurize(y, y_pred, y_aug)

        u_start = 1 + len(exm['augments'])
        u_end = u_start + len(exm['u_exms'])

        for start, end in zip([0, u_start], [u_start, u_end]):
            if end == start:
                continue
            # create distribution and sample
            # featurize
            # if start == 0 and end - start > 1:
            if end - start > 1:
                logits = policy(features[start:end], labeled=(start==0))

                probs = sharpen(logits, T=0.1)
                m = torch.distributions.categorical.Categorical(probs=probs)
                action = m.sample()
                # action = torch.LongTensor([random.randint(0, end-start-1)]).to(model.device)
                # action = torch.LongTensor([0]).to(model.device)
                # if ori_idx == 0:
                #     print(m.probs)
            else:
                action = 0

            # create indices
            new_idx = int(start + action + ori_idx)
            exm_indices.append(new_idx)
            new_batch.append(all_exms[new_idx])

            # get probability (with gradients)
            # if start == 0 and end - start > 1:
            # if start == 0 and end - start > 1:
            if end - start > 1:
                log_prob_sum += m.log_prob(action)

    # debug
    assert idx == len(all_exms)
    enc_indices = [aug_mp[k] for k in exm_indices]
    # print(ys[enc_indices][-1], y_preds[exm_indices][-1], new_batch[-1][5])

    return padder(new_batch), x_encs[enc_indices],\
           ys[enc_indices], log_prob_sum


def auto_ssl(model, batch,
             aug_enc=None,
             labels=None,
             policy=None,
             get_ind=False):
    """Train the model on a filtered batch with the weighting policy

    Args:
        model (MultiTaskNet): the model state
        batch (tuple): the input batch
        aug_enc (Tensor, optional): representations of the augmented examples
        labels (LongTensor, optional): the labels (real or guessed)
                           for computing the loss
        policy (AugmentPolicyNetV4, optional): the weighting policy
        get_ind (boolean, optional): return the weight if set True

    Returns:
        Tensor: the loss (of 0-d)
        Tensor: (if get_ind is True) the weight for each training instance
    """
    _, x, _, _, _, y, _, taskname = batch
    taskname = taskname[0]
    num_classes = model.num_classes

    if policy is None:
        # y: LongTensor
        logits, y, _ = model(x, y, task=taskname)
        if 'sts-b' not in taskname.lower():
            y = F.one_hot(y, num_classes).float()
        ind = torch.ones(x.size()[0],).to(model.device)
    else:
        # y: onehot Tensor for labeled/augmented examples
        #    or guessed labels for unlabeled examples
        y = labels
        batch_size = x.size()[0]

        # generate model predictions
        x_enc = model(x, task=taskname, get_enc=True)

        # MixDA
        beta = torch.distributions.beta.Beta(0.8, 0.8)
        lam = beta.sample()
        lam = torch.min(1 - lam, lam)
        x_enc = x_enc * lam + aug_enc * (1.0 - lam)

        # new logits with the updated encodings
        logits, y, _ = model(x_enc=x_enc, y=y, task=taskname)

        # the weighting policy model generate the weights of each example
        ind = policy(x, y, prediction=logits.softmax(dim=-1))

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
            loss = torch.sum(-y[:, 1:] * logits[:, 1:].log() * ind, dim=-1).mean()
        else:
            loss = (torch.sum(-y * logits.log(), -1) * ind).mean()

    if get_ind:
        return loss, ind.detach()
    else:
        return loss


def train(model,
          filter_model,
          weight_model,
          l_set,
          w_aug_set,
          s_aug_set,
          u_set,
          v_set,
          optimizer,
          filter_optimizer,
          weight_optimizer,
          scheduler=None,
          fp16=False,
          batch_size=32,
          no_ssl=False):
    """Perform one epoch of LM training with meta-learning

    Args:
        model (MultiTaskModel): the model state
        filter_model (FilterPolicyNet): the filtering model
        weight_model (AugmentPolicyNetV4): the weighting model
        l_set (SnippextDataset): the labeled set
        w_aug_set (SnippextDataset): the augmented labeled set (weak, less noise)
        s_aug_set (SnippextDataset): the augmented labeled set (strong, more noise)
        u_set (SnippextDataset): the unlabeled set
        v_set (SnippextDataset): the validation set
        optimizer (Optimizer): Adam
        filter_optimizer (Optimizer): Adam
        weight_optimizer (Optimizer): Adam
        scheduler (Optional): the linear rate schedule for the target model
        fp16 (boolean, Optional): whether to use fp16
        batch_size (int, Optional): batch size
        no_ssl (boolean, Optionial): if True, then ignore unlabeled data

    Returns:
        None
    """
    # validation batches
    da_batches = None
    # create the training batches
    batches = create_batches(l_set,
                             w_aug_set,
                             s_aug_set,
                             u_set,
                             batch_size=batch_size,
                             no_ssl=no_ssl)

    for i, batch in enumerate(batches):
        try:
            da_batch = next(da_batches)
        except:
            da_batches = iter(data.DataLoader(dataset=v_set,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=v_set.pad))
            da_batch = next(da_batches)

        # run the filtering model
        new_batch, aug_enc, new_y, log_prob = process_filter(model,
                                            filter_model,
                                            batch,
                                            l_set.pad)

        words, x, is_heads, tags, mask, y, seqlens, taskname = new_batch
        taskname = taskname[0]

        # phase one: update the policy parameters

        # make a copy of the current model values
        model_values = [p.data.clone() for p in model.parameters()]

        # learning rate
        lr = [group['lr'] for group in optimizer.param_groups]

        # virtual SGD step
        optimizer.zero_grad()
        train_loss = auto_ssl(model, new_batch, policy=weight_model,
                              aug_enc=aug_enc,
                              labels=new_y)
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
        val_loss = auto_ssl(model, da_batch, policy=None)
        if fp16:
            with amp.scale_loss(val_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            val_loss.backward()

        # compute epsilon
        model_gradients = [p.grad.data.clone() if p.grad is not None \
                else None for p in model.parameters()]
        with torch.no_grad():
            grad_L2 = torch.zeros(()).to(model.device)
            for g in model_gradients:
                if g is not None:
                    grad_L2 += g.pow(2).sum()

            epsilon = 0.01 / grad_L2.sqrt()

        # w_plus and w_minus
        weight_optimizer.zero_grad()
        for sign in [-1.0, 1.0]:
            with torch.no_grad():
                for p, g, v in zip(model.parameters(), model_gradients, model_values):
                    if g is not None:
                        p.data = v + sign * epsilon * g
                    else:
                        p.data = v
            loss_pm = auto_ssl(model, new_batch, policy=weight_model,
                               aug_enc=aug_enc,
                               labels=new_y)
            loss_pm = loss_pm * -sign / 2 / epsilon * lr[0]
            if fp16:
                with amp.scale_loss(loss_pm, weight_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_pm.backward()

        # optimize
        weight_optimizer.step()

        # reset the model's parameters
        with torch.no_grad():
            for p, v in zip(model.parameters(), model_values):
                p.data = v

        # phase two: update the model parameters
        optimizer.zero_grad()
        loss, ind = auto_ssl(model, new_batch,
                             policy=weight_model,
                             aug_enc=aug_enc,
                             labels=new_y,
                             get_ind=True)

        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # phase three: update the filter model parameters
        with torch.no_grad():
            v_loss = auto_ssl(model, da_batch, policy=None)

        reward = log_prob * -v_loss # flag
        if fp16:
            with amp.scale_loss(reward, filter_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            reward.backward()
        filter_optimizer.step()

        # filter zero grad
        filter_optimizer.zero_grad()

        # debug: v_loss should decrease
        # with torch.no_grad():
        #     v_loss_after = auto_ssl(model, da_batch, policy=None)
        # print("v_loss before: ", val_loss.detach().cpu().numpy(), "v_loss after: ", v_loss_after.detach().cpu().numpy())

        # # get ind stat
        # ind_stat += ind.view(4,-1).sum(dim=-1).cpu().numpy()
        # print(ind_stat)

        if i == 0:
            words, x, is_heads, tags, mask, y, seqlens, _ = new_batch

            print("=====sanity check======")
            print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", get_tokenizer().convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            y_sample = y.cpu().numpy()[0]
            if np.isscalar(y_sample):
                print("y:", y_sample, new_y[0].cpu())
            else:
                print("y:", y_sample[:seqlens[0]])

            unlabeled = new_y.max(dim=-1)[0].le(0.999).cpu()
            num_unlabeled = unlabeled.sum()
            print("num_unlabeled: ", int(num_unlabeled.cpu()))

            print("tags:", tags[0])
            print("mask:", mask[0])
            print("seqlen:", seqlens[0])
            print("task_name:", taskname)

            print("unlabeled_weight: ", (unlabeled.float() * ind.cpu()).sum() / num_unlabeled)
            print("labeled_weight: ", ((1 - unlabeled.float()) * ind.cpu()).sum() / (len(words) - num_unlabeled))
            print("filter model: ", filter_model.fc.weight.data, filter_model.fc.bias.data)

            max_idx = int(ind.argmax().cpu())
            min_idx = int(ind.argmin().cpu())
            print('max_ind: ', ind.max(), 'min_ind: ', ind.min())
            if 'tagging_' not in taskname:
                names = ['labeled', 'unlabeled']
                print('good (%s), y=%d: %s' % (names[unlabeled[max_idx]], int(y[max_idx]), words[max_idx]))
                print('bad (%s), y=%d: %s' % (names[unlabeled[min_idx]], int(y[min_idx]), words[min_idx]))

            print("=======================")

        if i%10 == 0: # monitoring
            print(f"step: {i}, task: {taskname}, loss: {loss.item()}, v_loss: {v_loss.item()}, reward: {reward.item()}")
            del loss


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
    padder = SnippextDataset.pad

    # iterators for dev/test set
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

    # the model to be trained
    model = MultiTaskNet([task_config], device=device,
                         lm=hp.lm, bert_path=hp.bert_path)

    # the model for filtering
    filter_model = FilterPolicyNet(num_classes=max(1, num_classes),
                                   device=device)

    # the model for weighting
    weight_model = AugmentPolicyNetV4(num_classes, device,
                         lm=hp.lm, bert_path=hp.bert_path)


    # move to device
    model = model.to(device)
    filter_model = filter_model.to(device)
    weight_model = weight_model.to(device)

    # construct the optimizers and schedulers
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    filter_optimizer = AdamW(filter_model.parameters(), lr=0.01) # 0.01
    weight_optimizer = AdamW(weight_model.parameters(), lr=hp.lr)

    if device == 'cuda' and hp.fp16:
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level='O2')
        filter_model, filter_optimizer = amp.initialize(filter_model,
                                          filter_optimizer,
                                          opt_level='O2')
        weight_model, weight_optimizer = amp.initialize(weight_model,
                                          weight_optimizer,
                                          opt_level='O2')

    # learning rate scheduler
    # half labeled and half unlabeled
    num_steps = len(l_set) * 2 // hp.batch_size * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                      num_warmup_steps=num_steps / 10,
                                      num_training_steps=num_steps)

    # create logging
    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    writer = SummaryWriter(log_dir=hp.logdir)

    best_dev_f1 = best_test_f1 = 0.0
    # train the model and policy
    for epoch in range(1, hp.n_epochs+1):
        train(model,
              filter_model,
              weight_model,
              l_set,
              w_aug_set,
              s_aug_set,
              u_set,
              validset,
              optimizer,
              filter_optimizer,
              weight_optimizer,
              scheduler=scheduler,
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

    writer.close()
