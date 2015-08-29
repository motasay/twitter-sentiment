import codecs, logging

from word2vec.word2vecReader import Word2Vec
from preprocessing import preprocess_tweet
from features import get_word2vec_features, NUM_LINGUISTIC_FEATURES, get_linguistic_features
from evaluation import print_evaluations
from nn import NN

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.linear_model import LogisticRegression

def load_word2vec(path='../models/word2vec_twitter_model.bin'):
    return Word2Vec.load_word2vec_format(path, binary=True)

def load_data(vec_function, num_features, num_test_samples_per_class=500):
    # first load the raw data
    f = codecs.open('../data/positive-all', 'r', 'utf-8')
    positive = {l.strip() for l in f}
    f.close()

    f = codecs.open('../data/negative-all', 'r', 'utf-8')
    negative = {l.strip() for l in f}
    f.close()

    f = codecs.open('../data/neutral-all', 'r', 'utf-8')
    neutral = {l.strip() for l in f}
    f.close()
    
    # convert the sentences to vectors
    positive_features = np.zeros((len(positive), num_features), dtype=np.float32)
    negative_features = np.zeros((len(negative), num_features), dtype=np.float32)
    neutral_features  = np.zeros((len(neutral) , num_features), dtype=np.float32)

    for i, sentence in enumerate(positive):
        sent_vec = vec_function(sentence)
        positive_features[i,] = sent_vec

    for i, sentence in enumerate(negative):
        sent_vec = vec_function(sentence)
        negative_features[i,] = sent_vec

    for i, sentence in enumerate(neutral):
        sent_vec = vec_function(sentence)
        neutral_features[i,] = sent_vec
    
    # finally split into train/test and combine them into one big matrix
    pos_train, pos_test = train_test_split(positive_features, test_size=num_test_samples_per_class, random_state=22)
    neg_train, neg_test = train_test_split(negative_features, test_size=num_test_samples_per_class, random_state=22)
    neu_train, neu_test = train_test_split(neutral_features , test_size=num_test_samples_per_class, random_state=22)

    X_train = np.vstack((
        pos_train,
        neg_train,
        neu_train
    ))
    X_test  = np.vstack((
        pos_test,
        neg_test,
        neu_test
    ))
    Y_train = np.hstack((
        np.ones((pos_train.shape[0]), dtype=np.float32),
        np.ones((neg_train.shape[0]), dtype=np.float32) * -1,
        np.zeros((neu_train.shape[0]), dtype=np.float32)
    ))
    Y_test = np.hstack((
        np.ones((pos_test.shape[0]), dtype=np.float32),
        np.ones((neg_test.shape[0]), dtype=np.float32) * -1,
        np.zeros((neu_test.shape[0]), dtype=np.float32)
    ))

    # shuffle 'em
    X_train, Y_train = shuffle(X_train, Y_train, random_state=111)
    X_test , Y_test  = shuffle(X_test , Y_test , random_state=111)
    
    return X_train, Y_train, X_test , Y_test

def run():
    np.random.seed(142)
    
    # build a dataset from the word2vec features
    logging.info('Loading the word2vec model...')
    w2v = load_word2vec()
    
    vec_function = lambda sentence: get_word2vec_features(w2v, sentence)
    num_features = w2v.layer1_size
    
    logging.info('Building the word2vec sentence features...')
    w2v_X_train, w2v_Y_train, w2v_X_test , w2v_Y_test = load_data(vec_function, num_features)
    
    # del w2v # don't need it anymore
    
    # now train a neural net on this dataset
    logging.info('Training a neural net on the word2vec features...')
    w2v_nn = NN(400, 1000, 300)
    w2v_nn.train(w2v_X_train, w2v_Y_train)
    
    # threshold the results and show some evaluations to see if we can improve on this
    w2v_train_predictions = w2v_nn.predict_classes(w2v_X_train)
    w2v_test_predictions  = w2v_nn.predict_classes(w2v_X_test)
    
    print_evaluations(w2v_Y_test, w2v_test_predictions)
    
    # Now let's get the linguistic features
    logging.info('Building the linguistic features...')
    
    vec_function = lambda sentence: get_linguistic_features(sentence)
    num_features = NUM_LINGUISTIC_FEATURES
    ling_X_train, ling_Y_train, ling_X_test , ling_Y_test = load_data(vec_function, num_features)
    
    # now let's combine the output of the neural net with the linguistic features
    # first binarize the neural net predictions so that we have one indicator feature per class
    # convert the outputs to 3 indicator (i.e. binary) features
    mlb = MultiLabelBinarizer()
    w2v_train_predictions_binarized = mlb.fit_transform(w2v_train_predictions.reshape(-1, 1))
    w2v_test_predictions_binarized  = mlb.fit_transform(w2v_test_predictions.reshape(-1, 1))
    
    # now stack these with the ling features
    # now combine the features and train a new classifier
    X_train = np.hstack((
        w2v_train_predictions_binarized,
        ling_X_train
    ))
    X_test = np.hstack((
        w2v_test_predictions_binarized,
        ling_X_test
    ))
    
    logging.info('Normalising the final dataset to unit length')
    lengths = np.linalg.norm(X_train, axis=1)
    X_train = X_train / lengths[:, None] # divides each row by the corresponding element
    lengths = np.linalg.norm(X_test, axis=1)
    X_test = X_test / lengths[:, None]
    
    logging.info('Training a neural net on the final dataset...')
    nn = NN(X_train.shape[1], 3000, 600)
    nn.train(X_train, w2v_Y_train)

    predictions = nn.predict_classes(X_test)
    print_evaluations(w2v_Y_test, predictions)
    
    predictions = nn.predict_continuous(X_test)
    print_evaluations(w2v_Y_test, predictions, classification=False)
    
    # logging.info('Training a logistic regression model on the final dataset...')
    # lr = LogisticRegression(C=1e5, class_weight='auto', random_state=33)
    # lr.fit(X_train, w2v_Y_train)
    #
    # predictions = lr.predict(X_test)
    # print_evaluations(w2v_Y_test, predictions)
    
    logging.info('Done.')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    run()