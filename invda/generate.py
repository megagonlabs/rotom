import torch
import sys
import os
import jsonlines
import argparse
import spacy

from tqdm import tqdm
from transformers import T5ForConditionalGeneration,T5Tokenizer

def set_seed(seed):
    """set seeds"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


nlp = spacy.load('en_core_web_sm')

def dump_output(fn, results):
    """Dump augmentations to a jsonline file
    """
    with jsonlines.open(fn + '.augment.jsonl', mode='w') as writer:
        for res in results:
            writer.write(res)


def generate(model, tokenizer, device, sentence):
    """Generate using a T5 seq2seq model

    Args:
        model: the T5 model state
        tokenizer: the T5 tokenizer
        device (str): cpu or cuda
        sentence (str): the input string

    Returns:
        List of str: the augmentations
    """
    text =  "corrupt: " + sentence + " </s>"
    max_len = 256

    encoding = tokenizer.encode_plus(text,
                                     max_length=max_len,
                                     truncation=True,
                                     pad_to_max_length=True)
    input_ids = torch.LongTensor([encoding["input_ids"]]).to(device)
    attention_masks = torch.LongTensor([encoding["attention_mask"]]).to(device)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    with torch.no_grad():
        beam_outputs = model.generate(
            input_ids=input_ids,
            # attention_mask=attention_masks,
            do_sample=True,
            max_length=256,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=50)

    has_output = False
    result = []
    used = set([sentence.lower()])
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True)
        sl = sent.lower()
        if sl not in used:
            has_output = True
            result.append(sent)
            used.add(sl)
    return result


def process_cls(fn):
    """Process all lines in a training file for textcls

    Args:
        fn (str): the training set path

    Returns:
        None
    """
    print(fn)
    if os.path.exists(fn + '.augment.jsonl'):
        return
    results = []
    for sid, line in enumerate(tqdm(open(fn))):
        # process at most 10000 entries
        if sid >= 10000:
            break
        LL = line.strip().split('\t')
        if len(LL) < 2:
            continue
        sentence, label = LL[0], LL[-1]
        res = generate(model, tokenizer, device, sentence)
        results.append({'sid': sid,
            'original': sentence,
            'augment': res,
            'label': label})

    dump_output(fn, results)


def process_em(fn, size=1000):
    """Process all lines in a training file for em or cleaning

    Args:
        fn (str): the training set path

    Returns:
        None
    """
    print(fn)
    if os.path.exists(fn + '.augment.jsonl'):
        return
    results = []
    for sid, line in enumerate(tqdm(open(fn))):
        if sid >= size:
            break
        LL = line.strip().split('\t')
        if len(LL) < 2:
            continue
        if len(LL) == 2 or LL[1].strip() == '':
            sentence, label = LL[0], LL[-1]
            res = generate(model, tokenizer, device, sentence)
        else:
            sentence, label = ' *** '.join(LL[:2]), LL[-1]
            res = generate(model, tokenizer, device, sentence)
            sentence = sentence.replace('***', '\t')
            res = [s.replace('***', '\t') for s in res if '***' in s]

        results.append({'sid': sid,
            'original': sentence,
            'augment': res,
            'label': label})

    dump_output(fn, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--model_path", type=str, default='t5_corrupt')
    parser.add_argument("--type", type=str, default='cls')
    hp = parser.parse_args()

    # optional: set seeds for reproducibility
    set_seed(42)

    # t5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(hp.model_path)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # handling two types of input format
    fn = hp.input
    if hp.type == 'cls':
        process_cls(fn)
    if hp.type == 'em':
        process_em(fn)
