# Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation 
## https://arxiv.org/abs/1406.1078
## EuiYul Song (20214426)

### File System
* ```main.py```: a main class that preprocess, train, and evaluate the model
* ```Vocab.py```: a class that tokenizes and builds vocabulary dictionaries of input and output phrases
* ```Encoder.py```: a RNN Encoder with GRU that encodes a sequence of symbols into a fixed-length vector representation 
* ```Decoder.py```: a RNN Decoder with GRU that decodes the representation into another sequence of symbols
* ```app.py```: a Flask website that enables users to run French to English translation against our model
* ```Makefile```: a command line automation

### Data
* ```data/eng-fra.txt```: an English to French dictionary (separated by "
\t") that is downloaded from [link](http://www.manythings.org/anki/fra-eng.zip) and refined

### Command
* ```make run```: preprocess, train, and evaluate the model
* ```make app```: run website that translate inputted French phrase to English on ```localhost: 5000```

### Result
