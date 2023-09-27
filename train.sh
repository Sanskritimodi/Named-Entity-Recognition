
#python3 -m stanza.models.common.convert_pretrain ~/stanza_resources/gu/pretrain/fasttext.pt ~/root/ft/cc.gu.300.vec 150000
python3 bio_to_json.py

python3 -m stanza.utils.training.run_ner gu_conll --scheme bioes


