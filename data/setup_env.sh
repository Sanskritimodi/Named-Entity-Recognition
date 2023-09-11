#!/bin/bash

# Set NERBASE and NER_DATA_DIR variables
export NERBASE=/u/nlp/data/ner/stanza
export NER_DATA_DIR=/nlp/scr/$USER/data/ner

# Check if . is in PYTHONPATH, and add it if not
if [[ ! ":$PYTHONPATH:" == *":."* ]]; then
    export PYTHONPATH=$PYTHONPATH:.
fi

# Print the updated environment variables
echo "NERBASE is set to: $NERBASE"
echo "NER_DATA_DIR is set to: $NER_DATA_DIR"
echo "PYTHONPATH is set to: $PYTHONPATH"

# Append the export commands to your startup script (~/.bashrc)
echo "export NERBASE=$NERBASE" >> ~/.bashrc
echo "export NER_DATA_DIR=$NER_DATA_DIR" >> ~/.bashrc
echo 'if [[ ! ":$PYTHONPATH:" == *":."* ]]; then' >> ~/.bashrc
echo '    export PYTHONPATH=$PYTHONPATH:.' >> ~/.bashrc
echo 'fi' >> ~/.bashrc

echo "Environment variables set and appended to ~/.bashrc"

#to use:
#chmod +x setup_env.sh
#./setup_env.sh
