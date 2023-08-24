import spacy

model_path = 'models/model-best'  
nlp = spacy.load(model_path)

eval_file_path = 'data/gu_eval.iob'  
with open(eval_file_path, 'r', encoding='utf-8') as file:
    eval_data = file.read()

eval_docs = [nlp(text) for text in eval_data.split('\n') if text.strip()]

output = []

for doc in eval_docs:
    for tok in doc:
        label=tok.ent_iob_
        if label != "0":
            ent = tok.ent_type_
            label = label + '-' + ent
        output.append("\t".join([str(tok), label]))

print("\n".join(output))


