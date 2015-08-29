import numpy as np
from nltk import word_tokenize

from preprocessing import preprocess_tweet

import codecs
import re

NUM_LINGUISTIC_FEATURES = 8

f = codecs.open('../data/positive_words.txt', 'r', 'utf-8')
positive_words = {w.strip() for w in f}
f.close()
f = codecs.open('../data/negative_words.txt', 'r', 'utf-8')
negative_words = {w.strip() for w in f}
f.close()

# something to start with
intensity_words = {'really', 'so', 'too', 'very'}
negation_words  = {'not', 'no'}

# helper regular expressions
repeated_chars_regex = re.compile(r'(.)\1{2,}') # finds repeated chars for 3 or more times
multi_exclamation_regex = re.compile(r'!{2,}')

def get_features(word2vec, sentence):
    return np.hstack((
        get_word2vec_features(word2vec, sentence),
        get_linguistic_features(sentence)
    ))

def get_word2vec_features(word2vec, sentence):
    return word2vec.get_sentence_vec(word_tokenize(sentence))

def get_linguistic_features(sentence):
    '''
    Features:
    1: count of positive
    2: count of negative
    3: count of intensity words: very good, so cool
    4: count of elongated words: greeeeat, sooo good
    5: has a question mark
    6: has exclamation mark
    7: has exclamation repeated more than once!!!
    8: contains hashtag
    
    If you're interested in the usefulness of each of these,
    the following are the coefficients for a logistic regression
    model for the negative, neutral, and positive classes respectively
    [[ 0.35346987  4.92481448  1.80200111  2.88740596  1.34519168 -0.06794115
       1.20213894 -4.38018904]
     [-2.13637061 -3.40506267 -2.38370639 -1.82043459  0.25270369 -2.41349426
      -3.46977753  0.79135405]
     [ 2.92176862 -1.57157761  1.50862341  1.11264888 -2.41054806  3.26814909
       1.841855    3.13502001]]
    
    So many interesting insights can be seen from these. e.g.
    see how the coefficient of the hashtag is so big and leans
    toward classifying the tweet as positive?
    '''
    features = [0.0 for _ in range(NUM_LINGUISTIC_FEATURES)]
    
    preprocessed = preprocess_tweet(sentence)
    
    positives_count = 0
    negatives_count = 0
    intensity_count = 0
    elongated_count = 0
    for i, w in enumerate(preprocessed):
        # first make sure there's no negation before the word
        # if there's one, then we'll flip the sentiment of the word
        prev_is_negation = False
        if i > 0 and preprocessed[i-1] in negation_words:
            prev_is_negation = True
        
        if w in positive_words:
            if prev_is_negation:
                negatives_count += 1
            else:
                positives_count += 1
        elif w in negative_words:
            if prev_is_negation:
                positives_count += 1
            else:
                negatives_count += 1
        
        if w in intensity_words:
            intensity_count += 1
        
        if is_elongated(w):
            elongated_count += 1
    
    features[0] = positives_count
    features[1] = negatives_count
    features[2] = intensity_count
    features[3] = elongated_count
    
    if '?' in sentence: features[4] = 1
    if '!' in sentence:
        features[5] = 1
        if multi_exclamation_regex.search(sentence):
            features[6] = 1
    if '|||HASHTAG|||' in preprocessed:
        features[7] = 1
    
    # normalise to unit length
    length = np.linalg.norm(features)
    if length != 0.0:
        features = [x/length for x in features]
    
    return np.asarray(features, dtype=np.float32)
        
def is_elongated(word):
    return repeated_chars_regex.search(word) is not None