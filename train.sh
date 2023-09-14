python3 -m stanza.models.common.convert_pretrain ~/stanza_resources/gu/pretrain/fasttext.pt ~/Documents/NAMED\ ENTITY\ RECOGNITION/STANZA_NER/cc.gu.300.vec 150000

python -m stanza.utils.training.run_ner gu_conll


