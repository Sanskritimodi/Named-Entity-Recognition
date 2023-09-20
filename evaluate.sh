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
	'ner_model_path': '/Users/sanskriti/Documents/NAMED ENTITY RECOGNITION/STANZA_NER/gu/ner/gu_conll_nertagger.pt',
        # Use pretokenized text as input and disable tokenization
	'tokenize_pretokenized': True,
    'download_method': None 
    
}
nlp = stanza.Pipeline(**config) # Initialize the pipeline using a configuration dict
doc = nlp("આપણી જોઇ શકીએ છીએ કે ડીડીયુજીવાઇ અંતર્ગત વીજળી આપવામાં આવેલા ગામોની વાર્ષિક વુદ્ધિ આરજીજીવીવાઇથી ખુબ જ ઓછી છે પીઢ અભિનેતા વિજુ ખોટેનું નિધન કોલકાતાની મેડિકલ કોલેજમાં લાગી ભીષણ આગ, 250 લોકોને સુરક્ષિત બહાર કઢાયા માનસિક તાણ ઘેરી લેશે. ભારત વિશ્ર્વનો સૌથી યુવા દેશ છે. બાઇબલ એ પણ બતાવે છે કે યહોવાહ સત્યના પરમેશ્વર છે. ત્રણ વખતથી મુખ્યમંત્રી બની આવતી શીલા દીક્ષિત રાજકારણના નવાનિશાળિયા અરવિંદ કેજરીવાલના સામે ભારે મતોથી હારી ગઇ. લાહોર કિલ્લાના મામલા માટે જવાબદાર ધ વોલ્ડ સિટી ઓફ લાહોર ઓથોરિટીએ આ ઘટના પર આશ્ચર્ય વ્યક્ત કર્યું હતું અને ઇદ બાદ તુરંત આ પ્રતિમાને ઠીક કરવાની વાત કરી હતી. આજુબાજુના કેટલાક ટ્રાફિક ચિહ્નો સાથે લીલા પર ટ્રાફિક લાઇટ મોટી ઝાંઝરી ભારત દેશના પશ્ચિમ ભાગમાં આવેલા ગુજરાત રાજ્યના મધ્ય ભાગમાં આવેલા મહીસાગર જિલ્લામાં આવેલા લુણાવાડા તાલુકાનું એક ગામ છે.") # Run the pipeline on the pretokenized input text
#print(doc) # Look at the result

for sentence in doc.sentences:
    for ent in sentence.ents:
        print(f"Entity: {ent.text}, Tag: {ent.type}")
