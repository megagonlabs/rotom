datasets = "ATIS,SNIPS,SST-2,SST-5,TREC".split(',')
result_path = 'results_textcls/'
sizes = [100, 300, 500]
lm = 'distilbert'
batch_size = 32
n_epochs = 30

import os
import time

# auto mixda
for dataset in datasets:
    for size in sizes:
        for da in ['None', 'edbt20', 'del', 'invda', 'auto_filter_weight_no_ssl', 'auto_filter_weight']:
            if 'no_ssl' in da:
                da = da.replace('_no_ssl', '')
                no_ssl = True
            else:
                no_ssl = False

            for run_id in range(5):
                start = time.time()
                cmd = """CUDA_VISIBLE_DEVICES=2 python train_any.py \
              --task %s_%d \
              --logdir %s/ \
              --finetuning \
              --batch_size %d \
              --lr 3e-5 \
              --fp16 \
              --lm %s \
              --n_epochs %d \
              --da %s \
              --run_id %d""" % (dataset, size, result_path, batch_size,
                      lm, n_epochs, da, run_id)
                if no_ssl:
                    cmd += ' --no_ssl'
                print(cmd)
                os.system(cmd)
