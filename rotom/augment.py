import json
import pickle
import random
import numpy as np
import math
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec


class Augmenter(object):
    """Data augmentation operator.

    Support both span and attribute level augmentation operators.
    """
    def __init__(self, idf_fn='data/wikitext-idf.dat'):
        # self.index = json.load(open(index_fn))
        if idf_fn is not None:
            self.idf_dict = pickle.load(open(idf_fn, 'rb'))
        else:
            self.idf_dict = None


    def augment(self, tokens, labels, op='del', args=[]):
        """ Performs data augmentation on a sequence of tokens

        The supported ops:
           ['del',
            'token_del_tfidf',
            'token_del',
            'shuffle',
            'token_repl',
            'token_repl_tfidf'
            'all']

        Args:
            tokens (list of strings): the input tokens
            labels (list of strings): the labels of the tokens
            op (str, optional): a string encoding of the operator to be applied

        Returns:
            list of strings: the augmented tokens
            list of strings: the augmented labels
        """
        tfidf = 'tfidf' in op
        if 'del' in op:
            if 'token_del' in op:
                pos1 = self.sample_position(tokens, labels, tfidf)
                if pos1 < 0:
                    return tokens, labels
                new_tokens = tokens[:pos1] + tokens[pos1+1:]
                new_labels = labels[:pos1] + labels[pos1+1:]
                # new_tokens = tokens[:pos1] + ['[PAD]'] + tokens[pos1+1:]
                # new_labels = labels[:pos1] + ['<PAD>'] + labels[pos1+1:]
            else:
                # insert padding to keep the length consistent
                # span_len = random.randint(1, 3)
                max_len = args[0] if len(args) > 0 else 2
                span_len = random.randint(1, max_len)
                pos1, pos2 = self.sample_span(tokens, labels, span_len=span_len)
                if pos1 < 0:
                    return tokens, labels
                new_tokens = tokens[:pos1] + tokens[pos2+1:]
                new_labels = tokens[:pos1] + labels[pos2+1:]
        elif 'insert' in op:
            pos1 = self.sample_position(tokens, labels, tfidf)
            if pos1 < 0:
                return tokens, labels
            ins_token = self.sample_token(tokens[pos1])
            if ins_token.lower() != tokens[pos1].lower():
                new_tokens = tokens[:pos1] + [ins_token] + tokens[pos1:]
                new_labels = labels[:pos1] + ['O'] + labels[pos1:]
            else:
                new_tokens, new_labels = tokens, labels
        elif 'repl' in op:
            pos1 = self.sample_position(tokens, labels, tfidf)
            if pos1 < 0:
                return tokens, labels
            ins_token = self.sample_token(tokens[pos1])
            new_tokens = tokens[:pos1] + [ins_token] + tokens[pos1+1:]
            new_labels = labels[:pos1] + ['O'] + labels[pos1+1:]
        elif 'shuffle' in op:
            max_len = args[0] if len(args) > 0 else 4
            span_len = random.randint(2, max_len)
            pos1, pos2 = self.sample_span(tokens, labels, span_len=span_len)
            if pos1 < 0:
                return tokens, labels
            sub_arr = tokens[pos1:pos2+1]
            random.shuffle(sub_arr)
            new_tokens = tokens[:pos1] + sub_arr + tokens[pos2+1:]
            new_labels = tokens[:pos1] + ['O'] * (pos2 - pos1 + 1) + labels[pos2+1:]
        else:
            new_tokens, new_labels = tokens, labels

        return new_tokens, new_labels


    def augment_sent(self, text, op='all', args=[]):
        """ Performs data augmentation on a classification example.

        Similar to augment(tokens, labels) but works for sentences
        or sentence-pairs.

        Args:
            text (str): the input sentence
            op (str, optional): a string encoding of the operator to be applied
            args (list, optional): the augmentation parameters (length, etc.)

        Returns:
            str: the augmented sentence
        """
        # 50% of chance of flipping
        # if ' [SEP] ' in text and random.randint(0, 1) == 0:
        #     left, right = text.split(' [SEP] ')
        #     text = right + ' [SEP] ' + left

        # tokenize the sentence
        if ' [SEP] ' in text:
            left, right = text.split(' [SEP] ')
            tokens = word_tokenize(left) + ['[SEP]'] + word_tokenize(right)
        else:
            tokens = word_tokenize(text)

        # avoid the special tokens
        labels = []
        for token in tokens:
            if token in ['[CLS]', '[SEP]']:
                labels.append('<SEP>')
            else:
                labels.append('O')

        if op == 'all':
            # RandAugment: https://arxiv.org/pdf/1909.13719.pdf
            N = 3
            ops = ['del',
            'token_del_tfidf',
            'token_del',
            'shuffle',
            'token_repl',
            'token_repl_tfidf']
            for op in random.choices(ops, k=N):
                tokens, labels = self.augment(tokens, labels, op=op)
        if op == 'corrupt' or op == 'corrupt_plus':
            # remove 20% of tokens
            N = len(tokens)
            to_change = N // 3
            for _ in range(to_change):
                tokens, labels = self.augment(tokens, labels, op='token_del_tfidf')
            N = len(tokens)
            to_change = N // 5
            for _ in range(to_change):
                tokens, labels = self.augment(tokens, labels, op='token_repl_tfidf')
            to_change = N // 5
            for _ in range(to_change):
                tokens, labels = self.augment(tokens, labels, op='shuffle')
        elif op == 'corrupt_minus':
            # add 20% of tokens
            N = len(tokens)
            to_change = N // 3
            for _ in range(to_change):
                tokens, labels = self.augment(tokens, labels, op='insert')
            N = len(tokens)
            to_change = N // 5
            for _ in range(to_change):
                tokens, labels = self.augment(tokens, labels, op='token_repl_tfidf')
            to_change = N // 5
            for _ in range(to_change):
                tokens, labels = self.augment(tokens, labels, op='shuffle')
        elif op[:6] == 'multi_':
            N = len(tokens)
            rep = 1 + N // 10
            new_op = op[6:]
            for _ in range(rep):
                tokens, labels = self.augment(tokens, labels, op=new_op)
        else:
            tokens, labels = self.augment(tokens, labels, op=op, args=args)
        results = ' '.join(tokens)
        return results

    def sample_span(self, tokens, labels, span_len=3):
        candidates = []
        for idx, token in enumerate(tokens):
            if idx + span_len - 1 < len(labels) and ''.join(labels[idx:idx+span_len]) == 'O'*span_len:
                candidates.append((idx, idx+span_len-1))
        if len(candidates) <= 0:
            return -1, -1
        return random.choice(candidates)

    def sample_position(self, tokens, labels, tfidf=False):
        # candidates = []
        # for idx, token in enumerate(tokens):
        #     if labels[idx] == 'O':
        #         candidates.append(idx)
        # if len(candidates) <= 0:
        #     return -1
        # return random.choice(candidates)

        candidates = []
        for idx, token in enumerate(tokens):
            if labels[idx] == 'O':
                candidates.append(idx)

        if len(candidates) <= 0:
            return -1
        if tfidf:
            oov_th = math.log(1e8)
            weight = {}
            max_weight = 0.0
            for idx, token in enumerate(tokens):
                token = token.lower()
                # if token not in index:
                #     continue
                if token not in self.idf_dict:
                    self.idf_dict[token] = oov_th
                if token not in weight:
                    weight[token] = 0.0
                # weight[token] += index[token]['idf']
                weight[token] += self.idf_dict[token]
                max_weight = max(max_weight, weight[token])

            weights = []
            for idx in candidates:
                weights.append(max_weight - weight[tokens[idx].lower()] + 1e-6)

            weights = np.array(weights) / sum(weights)

            return np.random.choice(candidates, 1, p=weights)[0]
        else:
            return random.choice(candidates)


    def sample_token(self, token, max_candidates=10):
        """ Randomly sample a token's similar token stored in the index

        Args:
            token (str): the input token
            same_length (bool, optional): whether the return token should have the same
                length in BERT
            max_candidates (int, optional): the maximal number of candidates
                to be sampled

        Returns:
            str: the sampled token (unchanged if the input is not in index)
        """
        token = token.lower()
        # index = self.index['token']
        # if token in index and index[token]['similar_words'] != None:
        #     candidates = [t for t, _ in \
        #                   index[token]['similar_words'][:max_candidates]]
        candidates = []
        syns = wordnet.synsets(token)
        for syn in syns:
            for lem in syn.lemmas():
                w = lem.name().lower()
                if w != token and w not in candidates and '_' not in w:
                    candidates.append(w)

        # print(candidates)
        if len(candidates) <= 0:
            return token
        else:
            return random.choice(candidates)


if __name__ == '__main__':
    ag = Augmenter()
    text = 'COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020 [SEP] COL content VAL sigmod conference 2010 papers 2019-12-31 COL year VAL 2019'
    for op in ['del',
               'shuffle',
               'insert',
               'all']:
        print(op)
        print(ag.augment_sent(text, op=op))
    ag = Augmenter(idf_fn='wikitext-idf.dat')
    text = 'demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal film with an emotional wallop'
    for op in ['token_repl', 'token_repl_tfidf', 'token_del', 'token_del_tfidf']:
        print(op)
        print(ag.augment_sent(text, op=op))

    text_list = ['adobe creative suite 3 ( cs3 ) design suite standard upgrade ( upsell ) mac',
            'advances in databases and information systems ( adbis )',
            'effective timestamping in relational databases',
            'tutorial : ldap directory services - just another database application?']

    for text in text_list:
        for op in ['token_del', 'shuffle']:
            print(op)
            print(ag.augment_sent(text, op=op))

    for op in ['corrupt_plus', 'corrupt_minus']:
        print(op)
        print(ag.augment_sent(text, op=op))

    # text = "it's a familiar story, but one that is presented with great sympathy"
    text = "Where is the Orange Bowl?"
    for op in ['corrupt_minus', 'corrupt_plus']:
        text = ag.augment_sent(text, op=op)
    print(text)
