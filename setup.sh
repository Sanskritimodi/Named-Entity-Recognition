#!/bin/bash

# Clone the Stanza repository
#git clone git@github.com:stanfordnlp/stanza.git

git clone https://github.com/stanfordnlp/stanza.git


# Change to the Stanza directory
cd stanza

# Switch to the 'dev' branch
git checkout dev

# Create a new branch 'bangla_ner'
git checkout -b guj_ner


#to use:
#chmod +x setup.sh
#./setup.sh
