# Datasets used in Rotom

The full list of datasets with sources, citations, and licenses:

|                 | Dataset     | Source Link                                                                            | Citation                                                         | License                                                          |
|-----------------|-------------|----------------------------------------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|
| Entity Matching | DeepMatcher | [[Link]](https://github.com/anhaidgroup/deepmatcher)                                             | [[1]](http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf) | BSD 3-clause                                                     |
| Error Detection | Raha        | [[Link]](https://github.com/BigDaMa/raha)                                                        | [[2]](http://raulcastrofernandez.com/papers/raha.pdf)                   | Apache 2.0                                                       |
| TextCLS         | AG News     | [[Link]](https://huggingface.co/datasets/ag_news)                                                | [[3]](https://dl.acm.org/doi/abs/10.1145/1060745.1060764)               | http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html |
|  TextCLS               | Amazon      | [[Link]](https://github.com/zhangxiangxiao/Crepe)                                                | [[4]](https://arxiv.org/abs/1509.01626)                                 | BSD 3-Clause                                                     |
|  TextCLS               | ATIS        | [[Link]](https://github.com/microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data) |                                                                  | MIT                                                              |
|  TextCLS               | SNIPS       | [[Link]](https://github.com/snipsco/snips-nlu)                                                   | [[5]](https://arxiv.org/pdf/1805.10190.pdf)                             | Apache 2.0                                                       |
| TextCLS                | SST         | [[Link]](https://nlp.stanford.edu/sentiment/index.html)                                          | [[6]](https://www.aclweb.org/anthology/D13-1170/)                       | GNU General Public License                                       |
| TextCLS                | TREC        | [[Link]](https://cogcomp.seas.upenn.edu/Data/QA/QC/)                                             | [[7]](https://www.aclweb.org/anthology/C02-1150.pdf)                    |                                                                  |
| TextCLS                | IMDB        | [[Link]](http://ai.stanford.edu/~amaas/data/sentiment/)                                          | [[8]](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)          | https://www.imdb.com/conditions                                  |

## Notes

* *Entity Matching:* Each dataset is a directory stored under ``em/``. Each dataset comes with a training, validation, and test sets called ``train.txt``, ``valid.txt`` and ``test.txt`` respectively. Both the clean and dirty versions of datasets are obtained from [DeepMatcher](https://github.com/anhaidgroup/deepmatcher).
* *Error Detection:* Each dataset is stored in a directory under ``cleaning/``. Each dataset comes in 4 sizes: [50, 100, 150, 200] and each size has 5 splits. For example, the 0-th split of the ``beers`` dataset of size 100 correspond to the directory ``cleaning/beers/100_10000/0/``. Each split contains a training, validation, a test set, and an unlabeled set named ``train.txt``, ``valid.txt``, ``test.txt``, and ``unlabeled.txt`` respectively.
* *Text Classification:* Each dataset is stored in a directory under ``textcls/``. The training and validation sets comes in size 100, 300, 500, or 1000 (e.g., ``train.txt.300``). The training file ``train.txt.full`` is used for semi-supervised learning. There is a single test set named ``test.txt``. 
* For each training set, we pre-computed the invda augmentations in the jsonlines files with suffix ``*.augment.jsonl``
