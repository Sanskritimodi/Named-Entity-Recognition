from stanza.utils.datasets.ner.utils import convert_bio_to_json

SHARDS = ('train', 'dev', 'test')

base_input_path="data/ner"
base_output_path="data/ner"
short_name="gu_conll"

trainFile = os.path.join(base_input_path, "gu_conll.train.bio")
testFile = os.path.join(base_input_path, "gu_conll.test.bio")
devFile = os.path.join(base_input_path, "gu_conll.dev.bio")

trainLines, testLines, devLines = utils.iob_dev_test_split(trainFile, 0.2, 0.2)

with open(trainFile, "w") as f:
    for sentence in trainLines:
        for line in sentence:
            f.write(line)
        f.write("\n")

with open(testFile, "w") as f:
    for sentence in testLines:
        for line in sentence:
            f.write(line)
        f.write("\n")

with open(devFile, "w") as f:
    for sentence in devLines:
        for line in sentence:
            f.write(line)
        f.write("\n")


convert_bio_to_json(base_input_path, base_output_path, short_name, suffix="bio", shard_names=SHARDS, shards=SHARDS)
