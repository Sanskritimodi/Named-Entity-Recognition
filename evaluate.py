import stanza

config = {
        # Comma-separated list of processors to use
	'processors': 'tokenize,ner',
        # Language code for the language to build the Pipeline in
	'lang': 'gu',
        # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
        # You only need model paths if you have a specific model outside of stanza_resources
	#'tokenize_model_path': './fr_gsd_models/fr_gsd_tokenizer.pt',
	#'mwt_model_path': './fr_gsd_models/fr_gsd_mwt_expander.pt',
	'ner_model_path': 'saved_models/ner/gu_conll_nertagger.pt',
        # Use pretokenized text as input and disable tokenization
	'tokenize_pretokenized': True,
	'download_method': None 
    
}
nlp = stanza.Pipeline(**config) # Initialize the pipeline using a configuration dict
input_file_path = 'data/gu_eval.iob'

# Read the contents of the file
with open(input_file_path, 'r', encoding='utf-8') as file:
	text = file.read()

'''def getWord(line):
	lineData = line.split()
	if len(lineData) > 0:
		return lineData[0]
	return line'''

#evalRecord = " ".join(map(getWord, evalData.split("\n")))
#evalRecord = " ".join(map(getWord, text.split("\n")))
evalRecord = "\n".join(text.split()) 

# Process the text from the file using the Stanza pipeline
doc = nlp(evalRecord)

for sentence in doc.sentences: #code for evaluation output in stanza ner
	for tok in sentence.tokens: 
		print("\t".join([tok.text, tok.ner]))
