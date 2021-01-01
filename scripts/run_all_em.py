datasets = """em_Amazon-Google
em_DBLP-ACM
em_DBLP-GoogleScholar
em_Walmart-Amazon
em_Abt-Buy""".split('\n')
# datasets = ['Textual/Abt-Buy']

result_path = 'results_em/'
sizes = [500, 1000]
lm = 'roberta'
batch_size = 32
n_epochs_list = [20, 20]

import os
import time

for da in ['del', 't5']:
    for dataset in datasets:
        for n_epochs, size in zip(n_epochs_list, sizes):
            for run_id in range(5):
                start = time.time()
                cmd = """CUDA_VISIBLE_DEVICES=2 python train_any.py \
              --task %s \
              --logdir %s/ \
              --finetuning \
              --batch_size %d \
              --lr 3e-5 \
              --fp16 \
              --lm %s \
              --n_epochs %d \
              --da %s \
              --size %d \
              --max_len 128 \
              --run_id %d""" % (dataset, result_path, batch_size,
                      lm, n_epochs, da, size, run_id)
                print(cmd)
                os.system(cmd)
