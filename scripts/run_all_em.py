datasets = """em_Amazon-Google
em_DBLP-ACM
em_DBLP-GoogleScholar
em_Walmart-Amazon
em_Abt-Buy
em_DBLP-ACM-dirty
em_DBLP-GoogleScholar-dirty
em_Walmart-Amazon-dirty""".split('\n')

result_path = 'results_em/'
sizes = [300, 450, 600, 750]
lm = 'roberta'
batch_size = 32
n_epochs_list = [20, 20, 20, 20]

import os
import time

for da in ['None', 'edbt20', 'del', 'invda', 'auto_ssl_no_ssl', 'auto_ssl']:
    if 'no_ssl' in da:
        da = da.replace('_no_ssl', '')
        no_ssl = True
    else:
        no_ssl = False

    for dataset in datasets:
        for n_epochs, size in zip(n_epochs_list, sizes):
            for run_id in range(5):
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
                if no_ssl:
                    cmd += ' --no_ssl'
                if 'auto_ssl' in da:
                    cmd += ' --balance'
                print(cmd)
                os.system(cmd)

# Note: use this hyperparameter set (no balancing but with warmup) for Walmart-Amazon and Walmart-Amazon-dirty
#
# CUDA_VISIBLE_DEVICES=2 python train_any.py \
# --task em_Walmart-Amazon-dirty \
# --logdir results_em_tmp/ \
# --finetuning \
# --batch_size 64 \
# --lr 3e-5 \
# --fp16 \
# --lm roberta \
# --n_epochs 20 \
# --da auto_filter_weight \
# --size 750 \
# --max_len 128 \
# --warmup \
# --run_id 0

# DM + RoBERTa
for dataset in datasets:
    dataset = dataset.replace('em_', '')
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
