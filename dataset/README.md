TaggerEva Dataset
=========
TaggerEva contains 13,000 method names sampled from 13 open-source Java projects and 3,000 natural language sentences sampled from [UD-GUM](https://gucorpling.org/gum/).


The columns are explained below:
1. ID: The number of ordered data item.
2. SEQUENCE: The token sequence. Tokens in both two type of data are splitted by the blank spacing.
3. POS: The corresponding part-of-speech tag sequence of the token sequence.
4. PROJECT (MN only): The source project of the MN.
5. FILE (MN only): The source file of the MN.

The dataset is splitted into three parts:

| Data | #Item | #Project |
|----|----|----|
|Train|9,000|9|
|Dev|1,000|1|
|Test(MNDataSet)|3,000|3|
|Test(NLDataSet)|3,000|-|

Adoption for taggers
------
For adoption to the special training/testing demand like command line interface, we transform the MNDataSet in several versions:
1. Stanford Format
> on/IN ready/JJ

The dataset has been transform to Stanford Format in the dir "stanford_format".

2. spaCy binary format: In the dir "spacy_format".
   
3. Flair format
> get VB
> 
> id NN

4. Ensemble format
    * TYPE: The return value's type of the method.
    * DECLARATION: Other parts of a method declaration.