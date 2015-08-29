from nltk import word_tokenize
from re import sub

f = open('../data/to_ignore.txt')
redundant_words = {w.strip() for w in f}
f.close()

def preprocess_tweet(tweet):
    assert isinstance(tweet, (str, unicode)), 'Got %s' % type(tweet)
    
    # lower case and strip white space
    tweet = tweet.lower().strip()
    
    # replace mentions with a special symbol RTs (e.g. RT TheFix: ...)
    tweet = sub(r'"?@{1}\S+:?', '|||MENTION|||', tweet)
    
    # replace hashtags with a special symbol
    tweet = sub(r'\s*#{1}\S+', ' |||HASHTAG|||', tweet)
    
    # replace numbers with a special symbol
    tweet = sub(r'(^|\s)\d+(\s|$)?', ' |||DIGIT||| ', tweet)

    # remove URLs
    tweet = sub(r'https?:\/\/\S+', '', tweet)
    
    # replace 've with have if it was preceded by a char
    tweet = sub(r'([a-zA-Z])+(\'ve)', r'\1 have', tweet)
    
    tweet = word_tokenize(tweet)
    
    # remove redundant words
    tweet = [w for w in tweet if w not in redundant_words]
    
    #replace n't with not
    tweet = [w if w != "n't" else "not" for w in tweet]
    
    return tweet