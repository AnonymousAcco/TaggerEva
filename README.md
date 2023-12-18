TaggerEva
========
### Install Selected Taggers
* [NLTK](https://www.nltk.org/install.html)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
* [spaCy](https://spacy.io/)
* [Flair](https://github.com/flairNLP/flair)
* [Ensemble*](https://github.com/SCANL/ensemble_tagger)

### Dataset
Please click [data](https://github.com/AnonymousAcco/TaggerEva/tree/main/dataset) to read the introduction of the TaggerEva dataset.

### Setup
1. Install the dependencies:
```sudo pip3 install -r requirements.txt```
   
2. Download the model of spaCy:
```python -m spacy download en_core_web_sm```

### Evaluation of Taggers
#### Origin Taggers
Run the evaluation.py using
```
python evaluation.py -m nl # NLDataSet
python evaluation.py -m id # MNDataSet
```
#### Retrained Taggers
Run the retrain.py using
```
python retrain.py -m eva 
```

#### Evaluation of Ensemble
1. Setup the tagger following its [documentation](https://github.com/SCANL/ensemble_tagger);
2. Copy the "eva_et.py" under "ensemble_tagger_implementation";
3. Run the script by
```
python eva_et.py
```
to get the output file "et_out.txt";
4. Copy "et_out.txt" back to TaggerEva and run "utils.py"

### Train the Taggers
#### NLTK & Flair
```
python retrain.py -m train
```
Can directly train NLTK and Flair.

#### Stanford & spaCy
These two taggers need to be trained by the command line interface. "train.py" can preprocess the dataset into their format.
```
python retrain.py -m pre
```
##### Stanford
* Copy the stanford_format data and "./model/stanford/maxnet.props" into the installation directory of Stanford CoreNLP.
* Run the command in command line:
```
java -mx1g -cp "*" edu.stanford.nlp.tagger.maxent.MaxentTagger -model "retrain_stanford.model" -testFile "stanford_test.txt" > stanford_out.txt
```
* Copy the "stanford_out.txt" back to TaggerEva

##### spaCy
 
  ```
  cd ./model/spacy/
  python -m spacy train spacy_config.cfg --paths.train ../dataset/spacy_format/train.spacy --paths.dev ../dataset/spacy_format/dev.spacy --output ./
```

After training, you can run the command:
```
python retrain.py -m eva
```
for evaluation.

### Model
The nltk, stanford and spacy retrained model has stored in "model". Due to the size limitation of Github, the flair model currently not been committed.

