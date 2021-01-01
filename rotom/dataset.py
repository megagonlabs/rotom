import numpy as np
import torch
import sys
import random
import jsonlines

from torch.utils import data
from .augment import Augmenter

sys.path.insert(0, "Snippext_public")
from snippext.dataset import SnippextDataset, get_tokenizer
from transformers.data import glue_processors
from transformers import InputExample

class BaseDataset(SnippextDataset):
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Return the ith item of in the dataset.

        Args:
            idx (int): the element index

        Returns:
            words, x, is_heads, tags, mask, y, seqlen, self.taskname
        """
        if self.augment_op is not None:
            return self.get(idx, [(self.augment_op, [])])
        else:
            return self.get(idx, [])

    def get(self, idx, ops=[], aug_idx=None):
        """Return the ith item of in the dataset and apply the transformation ops.

        The ops are of the form [(op_1, args_1), ..., (op_k, args_k)]

        Args:
            idx (int): the element index
            ops (list): the list of operators

        Returns:
            words, x, is_heads, tags, mask, y, seqlen, self.taskname
        """
        text_a = self.examples[idx].text_a
        text_b = self.examples[idx].text_b
        if text_b is not None and text_b.strip() == '':
            text_b = None
        label = self.examples[idx].label

        if len(ops) > 0 and ops[0][0] == 't5':
            examples = self.t5_examples[idx]
            if len(examples) > 0:
                if aug_idx is None:
                    e = random.choice(examples)
                else:
                    e = examples[aug_idx]
                text_a = e.text_a
                text_b = e.text_b
        elif len(ops) > 0:
            if text_b is not None:
                combined = text_a + ' [SEP] ' + text_b
            else:
                combined = text_a

            for op_args in ops:
                op, args = op_args
                combined = self.augmenter\
                    .augment_sent(combined, op, args=args)

            if text_b is not None:
                if ' [SEP] ' in combined:
                    text_a, text_b = combined.split(' [SEP] ')
            else:
                text_a = combined

        x = self.tokenizer.encode(text=text_a,
                text_pair=text_b,
                add_special_tokens=True,
                truncation=True,
                truncation_strategy='longest_first',
                max_length=self.max_len)

        if self.tag2idx is None:
            # regression
            y = float(label)
        else:
            if label in self.tag2idx:
                y = self.tag2idx[label] # label
            else:
                y = 0
        is_heads = [1] * len(x)
        mask = [1] * len(x)

        assert len(x)==len(mask)==len(is_heads), \
          f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"
        # seqlen
        seqlen = len(mask)

        # text for logging
        if text_b is None:
            log_text = text_a
        else:
            log_text = text_a + ' [SEP] ' + text_b

        return log_text, x, is_heads, label, mask,\
               y, seqlen, self.taskname


class TextCLSDataset(BaseDataset):
    def __init__(self, path, vocab, taskname, max_len=512,
            lm='distilbert', augment_op=None, size=None):
        self.taskname = taskname
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = get_tokenizer(lm=lm)
        # read path
        self.examples = []
        for uid, line in enumerate(open(path)):
            LL = line.strip().split('\t')
            if len(LL) == 2:
                e = InputExample(uid, LL[0], None, LL[1])
            elif len(LL) == 3:
                e = InputExample(uid, LL[0], LL[1], LL[2])
            self.examples.append(e)
            if size is not None and len(self.examples) >= size:
                break

        # vocab
        if None in self.vocab:
            # regression task
            self.tag2idx = self.idx2tag = None
        else:
            self.tag2idx = {tag: idx for idx, tag in enumerate(self.vocab)}
            self.idx2tag = {idx: tag for idx, tag in enumerate(self.vocab)}

        # augmentation
        self.augmenter = Augmenter()
        self.augment_op = augment_op

        # read the augment index
        if augment_op == 't5':
            self.t5_examples = []
            with jsonlines.open(path + '.augment.jsonl', mode='r') as reader:
                for row in reader:
                    exms = []
                    for sent in row['augment']:
                        LL = sent.split('\t')
                        if len(LL) == 0:
                            continue
                        label = row['label']
                        if len(LL) > 1:
                            e = InputExample(uid, LL[0], LL[1], label)
                        else:
                            e = InputExample(uid, LL[0], None, label)
                        exms.append(e)
                    self.t5_examples.append(exms)

