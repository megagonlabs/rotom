import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse

sys.path.insert(0, "Snippext_public")

from snippext.dataset import SnippextDataset, get_tokenizer
from torch.utils import data

from transformers import RobertaModel, DistilBertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from apex import amp


class DMDataset(SnippextDataset):
    def __init__(self,
                 path,
                 max_len=256,
                 size=256,
                 lm='roberta'):
        self.tokenizer = get_tokenizer(lm=lm)
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        for line in open(path):
            s1, s2, label = line.strip().split('\t')
            self.pairs.append((s1, s2))
            self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1 = self.tokenizer.encode(text=self.pairs[idx][0], max_length=self.max_len, truncation=True)
        x2 = self.tokenizer.encode(text=self.pairs[idx][1], max_length=self.max_len, truncation=True)
        x12 = self.tokenizer.encode(text=self.pairs[idx][0],
                                    text_pair=self.pairs[idx][1],
                   max_length=self.max_len, truncation=True)
        return x1, x2, x12, self.labels[idx]

    @staticmethod
    def pad(batch):
        x1, x2, x12, y = zip(*batch)

        maxlen = max([len(x) for x in x1+x2])
        x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
        x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]

        maxlen = max([len(x) for x in x12])
        x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]

        return torch.LongTensor(x1), \
               torch.LongTensor(x2), \
               torch.LongTensor(x12), \
               torch.LongTensor(y)


class DMModel(nn.Module):

    def __init__(self, device='cuda', lm='roberta'):
        super().__init__()
        if lm == 'roberta':
            self.bert = RobertaModel.from_pretrained('roberta-base')
        else:
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.device = device
        hidden_size = 768
        self.fc = torch.nn.Linear(hidden_size * 2, 2)
        # self.fc = torch.nn.Linear(hidden_size, 1)
        # self.cosine = nn.CosineSimilarity()
        # self.distance = nn.PairwiseDistance()

    def forward(self, x1, x2, x12):
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        x12 = x12.to(self.device)
        enc_pair = self.bert(x12)[0][:, 0, :]

        batch_size = len(x1)
        enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
        enc1 = enc[:batch_size]
        enc2 = enc[batch_size:]
        return self.fc(torch.cat((enc_pair, (enc1 - enc2).abs()), dim=1)) # .squeeze() # .sigmoid()

        # batch_size = len(x1)
        # enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
        # enc1 = enc[:batch_size]
        # enc2 = enc[batch_size:]
        # return self.distance(self.fc(enc1), self.fc(enc2)).sigmoid()
        # return self.fc((enc1 - enc2).abs()) # .squeeze() # .sigmoid()
        # return self.fc((enc1 - enc2).abs())

def evaluate(model, iterator, threshold=None):
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x1, x2, x12, y = batch
            logits = model(x1, x2, x12)
            # print(probs)
            probs = logits.softmax(dim=1)[:, 1]

            # print(logits)
            # pred = logits.argmax(dim=1)
            all_probs += probs.cpu().numpy().tolist()
            # all_p += pred.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.5, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th

def train_step(train_iter, model, optimizer, scheduler, hp):
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        x1, x2, x12, y = batch
        optimizer.zero_grad()
        prediction = model(x1, x2, x12)
        loss = criterion(prediction, y.to(model.device))
        # loss = criterion(prediction, y.float().to(model.device))
        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


def train(trainset, validset, testset, hp):
    padder = trainset.pad
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
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
    model = DMModel(device=device, lm=hp.lm)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging
    writer = SummaryWriter(log_dir=hp.logdir)

    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)

        # eval
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="DBLP-ACM")
    parser.add_argument("--logdir", type=str, default="results/")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--fp16", dest="fp16", action="store_true")

    hp = parser.parse_args()

    # set seed
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    path = 'data/em/%s' % hp.task
    train_path = os.path.join(path, 'train.txt')
    valid_path = os.path.join(path, 'train.txt') # ignore the validset
    test_path = os.path.join(path, 'test.txt')

    trainset = DMDataset(train_path, lm=hp.lm, size=hp.size)
    validset = DMDataset(valid_path, lm=hp.lm, size=hp.size)
    testset = DMDataset(test_path, lm=hp.lm)

    run_tag = 'em_%s_da=%s_id=%d_size=%d' % (hp.task, 'dm', hp.run_id, hp.size)
    train(trainset, validset, testset, hp)

