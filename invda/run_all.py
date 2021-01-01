import os

# EM
path = '../data/em'
datasets = ['Abt-Buy',
        'Amazon-Google',
        'DBLP-ACM',
        'DBLP-GoogleScholar',
        'Walmart-Amazon']

# train
for ds in datasets:
    cmd = """python train_t5.py \
  --data_dir %s/ \
  --train_filename train.txt \
  --valid_filename valid.txt \
  --model_output_dir em/%s/ \
  --gpu_list 2 --type em""" % (os.path.join(path, ds), ds)
    print(cmd)
    os.system(cmd)

# generate
for ds in datasets:
    cmd = """CUDA_VISIBLE_DEVICES=2 python generate.py \
        --input %s/train.txt \
        --model_path em/%s/ \
        --type em""" % (os.path.join(path, ds), ds)
    print(cmd)
    os.system(cmd)

#############################################################

# Cleaning
path = '../data/cleaning'
datasets = 'beers,hospital,movies,rayyan,tax'.split(',')
for ds in datasets:
    cmd = """python train_t5.py \
  --data_dir %s/100_10000/0/ \
  --train_filename unlabeled.txt \
  --valid_filename valid.txt \
  --model_output_dir cleaning/%s/ \
  --gpu_list 2 --type em""" % (os.path.join(path, ds), ds)
    print(cmd)
    os.system(cmd)

for ds in datasets:
    for size in [50, 100, 150, 200]:
        for run_id in range(5):
            cmd = """CUDA_VISIBLE_DEVICES=2 python generate.py \
                --input %s/%d_10000/%d/train.txt \
                --model_path cleaning/%s/ \
                --type em""" % (os.path.join(path, ds),
                        size, run_id, ds)
            print(cmd)
            os.system(cmd)

#############################################################

# TextCLS

import random

path = '../data/textcls'
datasets = ['AG', 'AMAZON2', 'AMAZON5', 'ATIS', 'SNIPS',
            'SST-2', 'SST-5', 'TREC']

# create 50,000 samples
all_lines = []
for ds in datasets:
    pa = os.path.join(path, ds, 'train.txt.full')
    all_lines += open(pa).readlines()[:10000]

random.shuffle(all_lines)
with open('textcls/train.txt', 'w') as fout:
    for line in all_lines[:50000]:
        fout.write(line)

with open('textcls/valid.txt', 'w') as fout:
    for line in all_lines[50000:60000]:
        fout.write(line)

# train
cmd = """python train_t5.py \
  --data_dir textcls/ \
  --train_filename train.txt \
  --valid_filename valid.txt \
  --model_output_dir textcls/ \
  --gpu_list 2 --type cls"""
print(cmd)
os.system(cmd)

# generate
for ds in datasets:
    cmd = """CUDA_VISIBLE_DEVICES=2 python generate.py \
        --input %s/train.txt \
        --model_path textcls/ \
        --type cls""" % (os.path.join(path, ds))
    print(cmd)
    os.system(cmd)
