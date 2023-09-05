from datasets import load_dataset

local_dataset_path = "/Users/sanskriti/Documents/NAMED ENTITY RECOGNITION/CONLL2003 ENG dataset/train_short.iob"

# Load the local dataset
raw_datasets = load_dataset("text", data_files={"train": local_dataset_path})

print(raw_datasets)
data_file_path = "/Users/sanskriti/Documents/NAMED ENTITY RECOGNITION/CONLL2003 ENG dataset/train_short.iob"
iob_data = []

def parse_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        item = {"tokens": [], "ner_tags": []}
        for line in f:
            line = line.strip()
            if line:
                token, label = line.split()
                item["tokens"].append(token)
                item["ner_tags"].append(label)
            iob_data.append(item)
        return iob_data
    
'''def parse_vertical_data(file_path):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        sentence = {"tokens": [], "ner_tags": []}
        for line in f:
            line = line.strip()
            if not line:
                if sentence["tokens"]:
                    sentences.append(sentence)
                sentence = {"tokens": [], "ner_tags": []}
            else:
                token, label = line.split()
                sentence["tokens"].append(token)
                sentence["ner_tags"].append(label)
    return sentences'''

sentences = parse_data(data_file_path)
from datasets import Dataset

#dataset = Dataset.from_dict({"tokens": [s["tokens"] for s in sentences], "ner_tags": [s["ner_tags"] for s in sentences]})

#raw_datasets = dataset

#raw_datasets
from datasets import Dataset, ClassLabel, DatasetDict, Sequence, Features, Value
# Create a ClassLabel for NER labels
#unique_ner_labels = set(ner_tag for example in sentences for ner_tag in example["ner_tags"])
unique_ner_labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
ner_label_class = Sequence(feature=ClassLabel(num_classes=len(unique_ner_labels), names=list(unique_ner_labels)))
#ner_label_class

raw_datasets = Dataset.from_dict({
    "tokens":[example["tokens"] for example in sentences],
    "ner_tags": [example["ner_tags"] for example in sentences]
},
features=Features({
        "tokens": Sequence(feature=Value("string")),
        "ner_tags": Sequence(feature=ClassLabel(num_classes=len(unique_ner_labels), names=list(unique_ner_labels)))
    }))
#raw_datasets
ner_feature = raw_datasets.features["ner_tags"]
#print(ner_feature)
label_names = ner_feature.feature.names
#label_names
from transformers import AutoTokenizer

#from transformers import AutoTokenizer, AutoModelForMaskedLM

#tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.is_fast
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
inputs = tokenizer(raw_datasets[0]["tokens"], is_split_into_words=True)
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels
labels = raw_datasets[0]["ner_tags"]
word_ids = inputs.word_ids()
#print(labels)
#print("\t")
#print(align_labels_with_tokens(labels, word_ids))
align_labels_with_tokens(labels, word_ids)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
  
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets.column_names,
)

'''Fine-tuning the model with the Trainer API'''
#data collation
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = data_collator([tokenized_datasets[i] for i in range(2)])
#batch["labels"]
#metrics
import evaluate

metric = evaluate.load("seqeval")

labels = raw_datasets[0]["ner_tags"]
labels = [label_names[i] for i in labels]

predictions = labels.copy()
predictions[2] = "O"
metric.compute(predictions=[predictions], references=[labels])


import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

#Defining the model
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

from huggingface_hub import login
login(token="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", add_to_git_credential=True)
from transformers import TrainingArguments

args = TrainingArguments(
    "huggingface-ner-model-en",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=True,
)
from transformers import Trainer

#trainer = Trainer(
    #model=model,
    #args=args,
    #train_dataset=tokenized_datasets,
    #eval_dataset=tokenized_datasets["validation"],
    #data_collator=data_collator,
    #compute_metrics=compute_metrics,
    #tokenizer=tokenizer,
#)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,  
    train_dataset=tokenized_datasets,  
    #eval_dataset=eval_dataset,    
)
trainer.train()
trainer.save_model()

trainer.push_to_hub(commit_message="Training complete")


model.push_to_hub()
