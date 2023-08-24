#!/bin/bash

echo "convert iob datafile to flair format"

python -m flair.datasets.convert_to_flair data/gu_train.iob data/

echo "training"

python -m flair.train -c config.cfg --cuda

echo "training completed"

from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ColumnCorpus

# define columns
columns = {0: 'text', 1: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = 'data/'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='gu_train.iob',
                              test_file='gu_eval.iob',
                              dev_file='gu_train.iob')
