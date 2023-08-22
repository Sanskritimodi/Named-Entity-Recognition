#!/bin/bash

echo "convert iob datafile to flair format"

python -m flair.datasets.convert_to_flair data/gu_train.iob data/

echo "training"

python -m flair.train -c config.cfg --cuda

echo "training completed"

