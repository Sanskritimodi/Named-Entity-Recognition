#!/bin/python

#
# Flair NER: Named Entity recognition using Flair library by Zalando Research.
# Algo module created for Acharya by Vimal Menon
# 2019
#

from typing import List
from pathlib import Path

from flair.data import Corpus, Sentence, Dictionary
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger, LanguageModel
from flair.trainers import ModelTrainer
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

class ner_trainer(object):
    def __init__(self, dataFolder, trainFile): 
        self.tag_type = 'ner'
        self.dataFolder = dataFolder
        self.trainFile = trainFile
        self.columns = {0: 'text', 1: 'pos', 2: 'empty', 3: 'ner'}
        pass

    def LoadConll03(self, dataFolder, trainFile, testFile=None, devFile=None):
        self.corpus = ColumnCorpus(dataFolder, self.columns, trainFile, testFile, devFile) 
        self.tag_dictionary = self.corpus.make_tag_dictionary(tag_type=self.tag_type)
        #print(self.corpus.train[0].to_tagged_string('ner'))
        print(self.tag_dictionary.idx2item)

    def LoadEmbeddings(self, emedding_types):
        self.embedding_types = emedding_types
        self.embeddings = StackedEmbeddings(embeddings=self.embedding_types)

    def train(self, model_path, 
            learning_rate=0.1, 
            batch_size=32, 
            epochs=150, 
            hidden_size=256, 
            use_crf=True):
        self.tagger = SequenceTagger(hidden_size=hidden_size,
                                    embeddings=self.embeddings,
                                    tag_dictionary=self.tag_dictionary,
                                    tag_type=self.tag_type,
                                    use_crf=use_crf)

        self.trainer = ModelTrainer(self.tagger, self.corpus)

        self.trainer.train(model_path, 
                            learning_rate=learning_rate, 
                            mini_batch_size=batch_size, 
                            max_epochs=epochs)


class ner_parser(object):
    def __init__(self):
        pass
    
    def loadModel(self, model_path):
        self.model = SequenceTagger.load(model_path)

    def oldparseSentence(self, instr):
        sentence = Sentence(instr)
        self.model.predict(sentence)
        return sentence.to_dict(tag_type='ner')

    def parseSentence(self, instr):
        sentence = Sentence(instr)
        self.model.predict(sentence)
        outdata = []
        for token in sentence.tokens:
            #strdata = token.text + " " + token.get_tag('ner').value
            strdata = token.text + " " + token.get_label('ner').value #get_tag() -> get_label()
            outdata+=[strdata]
        return "\n".join(outdata)

class trainLanguage(object):
    def __init__(self, charPath, is_forward=True):
        self.is_forward_lm = is_forward
        self.dictionary : Dictionary = Dictionary.load(charPath)

    def trainLanguage(self, corpusPath):
        self.corpus = TextCorpus(Path(corpusPath),
        self.dictionary,
        self.is_forward_lm,
        character_level=True)

        self.language_model = LanguageModel(self.dictionary,
        self.is_forward_lm,
        hidden_size=128,
        nlayers=10)

        self.trainer = LanguageModelTrainer(self.language_model, self.corpus)

        self.trainer.train('resources/taggers/language_model',
                    sequence_length=10,
                    mini_batch_size=10,
                    max_epochs=10)
