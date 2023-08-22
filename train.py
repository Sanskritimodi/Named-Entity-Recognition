import spacy

file_path = 'data/gu_train.iob'

with open(file_path, 'r', encoding='utf-8') as file:
    train_data = file.read()

!python3 -m spacy convert data/gu_train.iob data/ -c iob

!python -m spacy init fill-config ./base_config.cfg ./config.cfg

!python3 -m spacy train config.cfg --output models/ --paths.train ./data/gu_train.spacy --paths.dev ./data/gu_train.spacy 
