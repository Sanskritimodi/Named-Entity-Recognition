from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerDocumentEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Define columns
columns = {0: 'text', 1: 'ner'}

# Folder containing train, test, and dev files
data_folder = 'data/'

# Initialize corpus using column format and dataset files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='en_train.iob',)

# Define tag type
label_type = 'ner'

# Create tag dictionary
label_dictionary = corpus.make_label_dictionary(label_type=label_type)

''' Load IndicBERT Gujarati pretrained model
#indicbert_model_name = "l3cube-pune/gujarati-bert"
#tokenizer = AutoTokenizer.from_pretrained(indicbert_model_name)
#model = AutoModelForMaskedLM.from_pretrained(indicbert_model_name)

# Load model directly
#tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")'''

embedding_types = [
    WordEmbeddings('glove'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

'''Create transformer-based embedding using IndicBERT model
#from flair.embeddings import TransformerWordEmbeddings

# init embedding
#embedding = TransformerWordEmbeddings(indicbert_model_name)

# Initialize sequence tagger with transformer-based embedding
tagger = SequenceTagger(hidden_size=256,
                        embeddings=StackedEmbeddings(embeddings=[embedding]),
                        tag_dictionary=label_dictionary,
                        tag_type=label_type)'''

tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dictionary,
                        tag_type=label_type)

import time
for i in ['—','\\','|','/']*10:
     time.sleep(0.2)
     print(f'\rLoading... {i}', end="\r")

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/ner-english',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=20)

'''# Initialize trainer
trainer = ModelTrainer(tagger, corpus)



# Start training
trainer.train('resources/taggers/ner-english',
              train_with_dev=False,
              max_epochs=20)'''





