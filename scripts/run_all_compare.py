# datasets = "SST-5".split(',')
datasets = "IMDB,SST-5,TREC,SNIPS,SST-2,TREC".split(',')
cid_list = [1,1,1,2,2,2]
result_path = 'results_textcls_20epoch/'
lm = 'bert'
batch_size = 32

import os
import time

# auto mixda
for dataset, cid in zip(datasets, cid_list):
    for da in ['None', 'token_repl_tfidf', 'invda', 'auto_filter_weight_no_ssl']:
        if '_no_ssl' in da:
            da = da.replace('_no_ssl', '')
            no_ssl = True
        else:
            no_ssl = False

        if 'auto_' not in da and cid == 2:
            n_epochs = 20
        else:
            n_epochs = 10

        for run_id in range(5):
            start = time.time()
            cmd = """CUDA_VISIBLE_DEVICES=0 python train_any.py \
          --task compare%d_%s_%d \
          --logdir %s/ \
          --finetuning \
          --batch_size %d \
          --lr 3e-5 \
          --fp16 \
          --lm %s \
          --n_epochs %d \
          --da %s \
          --max_len 128 \
          --run_id %d""" % (cid, dataset, run_id, result_path, batch_size,
                  lm, n_epochs, da, run_id)
            if no_ssl:
                cmd += ' --no_ssl'
            print(cmd)
            os.system(cmd)
