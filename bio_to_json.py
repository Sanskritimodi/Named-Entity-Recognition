from stanza.utils.datasets.ner.utils import convert_bio_to_json

SHARDS = ('train', 'dev', 'test')

base_input_path="/data/ner"
base_output_path="/data/ner"
short_name="gu_conll"

convert_bio_to_json(base_input_path, base_output_path, short_name, suffix="bio", shard_names=SHARDS, shards=SHARDS)
