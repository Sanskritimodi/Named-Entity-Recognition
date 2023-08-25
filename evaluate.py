
import sys
import os
import logging
from flairNer import ner_parser


models_path = "resources/taggers/ner-gujarati/final-model.pt"

logging.getLogger("flair").setLevel(logging.WARNING)

logpath = "parse_ner.log"

logging.basicConfig(level=logging.DEBUG,
        filename=logpath,
        filemode='w')

PY = "PID:%d:PY::" % os.getpid()

logging.info(PY + "Starting ...")

np = ner_parser()
np.loadModel(models_path)
while True:
    inb = sys.stdin.buffer.read()
    logging.info(PY + "2:After reading bytes..\n")
    ins = inb.decode("utf-8")
    logging.info(PY + "3:Received data: %s len: %d", ins, len(ins))

    new_ins = " ".join(ins.split("\n"))

    tags = np.parseSentence(new_ins)
    logging.info(PY + "4: After parse..")
    print(tags)
    logging.info(PY + "5: Done")
    break
