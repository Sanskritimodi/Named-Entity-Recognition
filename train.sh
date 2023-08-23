python3 -m spacy convert data/gu_train.iob data/ -c iob

python -m spacy init fill-config ./base_config.cfg ./config.cfg

python3 -m spacy train config.cfg --output models/ --paths.train ./data/gu_train.spacy --paths.dev ./data/gu_train.spacy 
