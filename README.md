# Rotom
Code for the paper "Rotom: A Meta-Learned Data Augmentation Framework for Entity Matching, Data Cleaning, Text Classification, and Beyond"

## Requirements

* Python 3.7.7
* PyTorch 1.4.0
* Transformers 3.1.0
* NLTK (stopwords, wordnet)
* NVIDIA Apex (fp16 training)

Install required packages
```
pip install -r requirements.txt
git clone https://github.com/NVIDIA/apex.git
pip install -v --disable-pip-version-check --no-cache-dir ./apex
```

Install [Snippext](https://github.com/rit-git/Snippext_public):
```
git clone -b v1.0 https://github.com/rit-git/Snippext_public.git
cp -r Snippext_public/snippext .
```

## Model Training

To train a model with Rotom:
```
CUDA_VISIBLE_DEVICES=0 python train_any.py \
  --task em_DBLP-ACM \
  --size 300 \
  --logdir results_em/ \
  --finetuning \
  --batch_size 32 \
  --lr 3e-5 \
  --n_epochs 20 \
  --max_len 128 \
  --fp16 \
  --lm roberta \
  --da auto_filter_weight \
  --balance \
  --run_id 0
```

The current version supports 3 tasks: entity matching (EM), error detection (EDT), and text classification (TextCLS). The supported tasks are:
| Type    | Dataset Names                                                        | taskname pattern                         |
|---------|----------------------------------------------------------------------|------------------------------------------|
| EM      | Abt-Buy, Amazon-Google, DBLP-ACM, DBLP-GoogleScholar, Walmart-Amazon | em_{dataset}, e.g., em_DBLP-ACM          |
| EDT     | beers, hospital, movies, rayyan, tax                                 | cleaning_{dataset}, e.g., cleaning_beers |
| TextCLS | AG, AMAZON2, AMAZON5, ATIS, IMDB, SNIPS, SST-2, SST-5, TREC          | textcls_{dataset}, e.g., textcls_AG      |
| TextCLS, splits from [Hu et al.](https://arxiv.org/pdf/1910.12795.pdf) | IMDB, SST-5, TREC | compare1_{dataset}, e.g., compare1_IMDB |
| TextCLS, splits from [Kumar et al.](https://arxiv.org/pdf/2003.02245.pdf) | ATIS, SST-2, TREC | compare2_{dataset}, e.g., compare2_ATIS |

Parameters:
* ``--task``: the taskname pattern specified following the above table
* ``--size``: the dataset size (optional). If not specified, the entire dataset will be used. The size ranges are {300, 450, 600, 750} for EM, {50, 100, 150, 200} For EDT, and {100, 300, 500} for TextCLS
* ``--logdir``: the path for TensorBoard logging (F1, acc, precision, and recall)
* ``--finetuning``: always keep this flag on
* ``--batch_size``, ``--lr``, ``--max_len``, ``--n_epochs``: the batch size, learning rate, max sequence length, and the number of epochs for model training
* ``--fp16``: whether to use half-precision for training
* ``--lm``: the language model to fine-tune. We currently support bert, distilbert, and roberta
* ``--balance``: a special option for binary classification (EM and EDT) with skewed labels (#positive labels >> #negative labels). If this flag is on, then the training process will up-sample the positive labels
* ``--warmup``: (new) if this flag is on with SSL, then first warm up the model by training it on labeled data only before running SSL. Only support EM for now.
* ``--run_id``: the integer ID of the run e.g., {0, 1, 2, ...}
* ``--da``: the data augmentation method (See table below)

|                Method                |                                                                              Operator Name(s)                                                                             |
|:------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     No DA (simply LM fine-tuning)    |                                                                                    None                                                                                   |
|    Regular transformation-based DA   | ``['del', 'drop_col', 'append_col', 'swap', 'ins']`` for EM/EDT <br> ``['del', 'token_del_tfidf', 'token_del', 'shuffle', 'token_repl', 'token_repl_tfidf']`` for TextCLS |
|          Inversed DA (InvDA)         |                                                                                 t5 / invda                                                                                |
| Rotom (w/o semi-supervised learning) |                                                                         auto_filter_weight_no_ssl                                                                         |
|  Rotom (w. semi-supervised learning) |                                                                             auto_filter_weight                                                                            |

For the invda fine-tuning, see ``invda/README.md``.


## Experiment scripts

All experiment scripts are available in ``scripts/``. To run the experiments for a task (em, cleaning, or textcls):
```
python scripts/run_all_em.py
```
