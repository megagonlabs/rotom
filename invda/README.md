## InvDA

InvDA is the seq2seq-based data augmentation operator. InvDA takes as input an unlabeled training set and fine-tunes a T5 seq2seq model by corrupting and reconstructing the input sequences. The corruption is done by a combination of regular DA operators (e.g., token deletion). The T5 training and generation code is adapted from this [repo](https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-).

### Training the seq2seq model

To train the seq2seq model:
```
python train_t5.py \
  --data_dir ../data/em/Abt-Buy \
  --train_filename train.txt \
  --valid_filename valid.txt \
  --model_output_dir em/Abt-Buy/ \
  --gpu_list 2 \
  --type em
```

Parameters:
* ``--data_dir``: the dataset path containing the training files
* ``--train_filename``, ``--valid_filename``: the name of the training and validation files
* ``--model_output_dir``: the output directory of the model
* ``--gpu_list``: which GPU(s) to be used (e.g., "0", "1", "0,1")
* ``--type``: the type of the task. We currently support ``em`` (entity matching), ``cleaning`` (error detection), and ``textcls`` (text classification)

The script will fine-tune the model and store the model under the ``model_output_dir`` path.

### DA Generation with a trained model

To run the generation code:
```
CUDA_VISIBLE_DEVICES=2 python generate.py \
  --input ../data/em/Abt-Buy/train.txt \
  --model_path em/Abt-Buy/ \
  --type em
```

Parameters:
* ``--input``: the path to the training set that needs to be augmented
* ``--model_path``: the path to the trained seq2seq model
* ``--type``: the type of the task (should be the same as above). 

Running the above command will generate a jsonline file named ``train.txt.augment.jsonl`` (if not already computed) containing a list of augmentations for each example in the input file.


To run all the training and generation for all the datasets, simply use the ``run_all.py`` script:
```
python run_all.py
```

### Download the trained seq2seq models

Coming soon in the official release.
