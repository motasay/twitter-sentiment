# Description

This is a Tweet sentiment analyser that uses:

1. [A word2vec model][1] that was trained on 400 million Tweets.

2. (Very) simple linguistic features.

The system first trains a neural net (NN) on the word2vec vectors, then it combines the prediction of this NN with the linguistic features and finally trains another NN and print out some metrics of the final predictions.

[1]: Multimedia Lab @ ACL W-NUT NER Shared Task: Named Entity Recognition for Twitter Microposts using Distributed Word Representations

# Dependencies

The usual Python scientific stack (numpy, sklearn, etc.) is needed. In addition, nolearn is used, which in turn needs lasagne and theano to be installed. For a detailed list checkout the requirements.txt file.

# How to Use

The script expects to find the aforementioned word2vec model in a directory called models located at the root. Also the script expects to find three text files in the data folder:

1. negative-all: each line contains a tweet with a negative sentiment.

2. positive-all: each line contains a tweet with a positive sentiment.

3. neutral-all: each line contains a tweet with a neutral sentiment.

Once all the data are in place and the dependencies are installed, simply run `main.py` to see how the system does. The running time using the current NN architectures is about ~25 minutes on my machine.

# Possible Enhancements

1. The main enhacement is definitely the linguistic features. The ones used barely improve (if any) the performance. POS tags and other NLP features will definitely help. If we can find features that help in discriminating between the positive and neutral cases that would be great, as most of the confusion is between them (check the ipython notebook for more).

2. The preprocessing can also be improved. For example one could use the same preprocessing done for the word2vec model to get better features from word2vec. And I'm also sure we can improve it for the linguistic feature extraction in someway.