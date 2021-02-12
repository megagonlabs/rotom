datasets = "beers,flights,hospital,movies,rayyan,tax".split(',')
result_path = 'results_cleaning/'
sizes = [50, 100, 150, 200]
lm = 'roberta'
batch_size = 32

import os
import time

for dataset in datasets:
    for size in sizes:
        for da in ['None', 'swap', 'invda', 'auto_filter_weight_no_ssl', 'auto_filter_weight']:
            no_ssl = 'no_ssl' in da
            da = da.replace('_no_ssl', '')

            if size == 50:
                n_epochs = 30
            else:
                n_epochs = 20

            for run_id in range(5):
                cmd = """CUDA_VISIBLE_DEVICES=2 python train_any.py \
              --task cleaning_%s_%d_%d \
              --logdir %s/ \
              --finetuning \
              --batch_size %d \
              --lr 3e-5 \
              --fp16 \
              --lm %s \
              --max_len 128 \
              --n_epochs %d \
              --da %s \
              --run_id %d""" % (dataset, size, run_id, result_path, batch_size,
                      lm, n_epochs, da, run_id)
                if no_ssl:
                    cmd += ' --no_ssl'
                print(cmd)
                os.system(cmd)
