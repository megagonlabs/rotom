# datasets = """Amazon-Google
# DBLP-ACM
# DBLP-GoogleScholar
# Walmart-Amazon
# Abt-Buy""".split('\n')
datasets = """DBLP-ACM-dirty
DBLP-GoogleScholar-dirty
Walmart-Amazon-dirty""".split('\n')

result_path = 'results_dm/'
sizes = [750]
lm = 'roberta'
batch_size = 32
n_epochs_list = [20]

import os
import time

for dataset in datasets:
    for n_epochs, size in zip(n_epochs_list, sizes):
        for run_id in range(5):
            cmd = """CUDA_VISIBLE_DEVICES=2 python ditto/dm.py \
          --task %s \
          --logdir %s/ \
          --batch_size %d \
          --lr 3e-5 \
          --fp16 \
          --lm %s \
          --n_epochs %d \
          --size %d \
          --max_len 128 \
          --run_id %d""" % (dataset, result_path, batch_size,
                  lm, n_epochs, size, run_id)
            print(cmd)
            os.system(cmd)
