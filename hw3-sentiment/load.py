# Homework 3 - load.py
# Alex Pine
# 2015/02/15

import collections
import os
import numpy as np
import pickle
import random

'''
Note: No obligation to use this code, though you may if you like.  Skeleton code is just a hint for people who are not familiar with text processing in python. 
It is not necessary to follow. 
'''

PICKLE_FILE_PATH = '/tmp/hw3_pickled_reviews.p'
POS_REVIEWS_PATH = '/Users/pinesol/mlcs/hw3-sentiment/data/pos'
NEG_REVIEWS_PATH = '/Users/pinesol/mlcs/hw3-sentiment/data/neg'

def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    reviews = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        reviews.append(r)
    return reviews

def read_data(file):
    '''
    Read each file into a list of strings. 
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on', 
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(None, symbols).strip(), lines)
    words = filter(None, words)
    return words
	

def shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    pos_path = POS_REVIEWS_PATH
    neg_path = NEG_REVIEWS_PATH
	
    pos_reviews = folder_list(pos_path, 1)
    neg_reviews = folder_list(neg_path, -1)
	
    reviews = pos_reviews + neg_reviews
    random.shuffle(reviews)
    return reviews


def SplitAndSparsifyXY(shuffled_labeled_reviews):
    '''Code for question 3.1.
    Splits output of shuffle_data into X and Y matrices.
    Args:
      Matrix that is the output of shuffle_data()
    Returns: 
      X: list of collections.Counter objects.
      y: list of -1 or 1 ints.
    '''
    X = []
    y = []
    for labeled_review_list in shuffled_labeled_reviews:
        X.append(collections.Counter(labeled_review_list[:-1]))
        y.append(labeled_review_list[-1:][0])
    return X, y


def AddPositiveNegativeWordFeatures(X, y):
    '''Problem 6.1.
    Adds two new features to the training data matrix X: the number of 
    "positive" words and the number of "negative" words. Whether a word is
    "positive" or "negative" is determined by how often a word is mentioned
    in a positive or negative review, respectively.
    '''
    overall_word_counts = collections.Counter()
    positive_word_counts = collections.Counter()
    for x_row, y_val in zip(X, y):
        words = set(x_row.keys())
        overall_word_counts.update(words)
        if y_val > 0:
            positive_word_counts.update(words)
    RATIO_THRESHOLD = 0.9
    MIN_COUNT = 4
    positive_words = set([word for word, overall_count in overall_word_counts.iteritems()
                          if overall_word_counts[word] >= MIN_COUNT 
                          and 1.0*positive_word_counts[word]/overall_count >= RATIO_THRESHOLD])
    negative_words = set([word for word, overall_count in overall_word_counts.iteritems()
                          if overall_word_counts[word] >= MIN_COUNT 
                          and 1.0*(overall_count-positive_word_counts[word])/overall_count >= RATIO_THRESHOLD])
    print 'Number of different "positive" words:', len(positive_words)
    print 'Number of different "negative" words:', len(negative_words)
#    print 'positive words:', sorted(positive_words)
#    print 'negative words:', sorted(negative_words)

    perc_nonzero = 0.0
    for x_row in X:
        x_row['num_positive_words'] = len([word for word in x_row.iterkeys() 
                                           if word in positive_words])
        x_row['num_negative_words'] = len([word for word in x_row.iterkeys()
                                           if word in negative_words])
        if x_row['num_positive_words'] > 0 or x_row['num_negative_words'] > 0:
            perc_nonzero += 1.0
    perc_nonzero = 100.0 * perc_nonzero / len(X)
    print 'Percentage with a num_positive_words or num_negative_words score:', perc_nonzero


def PartitionData(X, y):
    '''Code for question 2.1.'''
    NUM_TRAINING_ROWS = 1500
    X_training = X[:NUM_TRAINING_ROWS]
    y_training = y[:NUM_TRAINING_ROWS]
    X_testing = X[NUM_TRAINING_ROWS:]
    y_testing = y[NUM_TRAINING_ROWS:]
    assert len(X_training) + len(X_testing) == 2000
    assert len(y_training) + len(y_testing) == 2000
    return X_training, y_training, X_testing, y_testing
	

def LoadData(add_extra_features=False):
    '''Code for question 2.1 and 3.1.'''
    if not os.path.isfile(PICKLE_FILE_PATH):
        print 'Pickle file not found, loading raw data...'
        shuffled_labeled_reviews = shuffle_data()
        pickle.dump(shuffled_labeled_reviews, open(PICKLE_FILE_PATH, 'wb'))
    else:
        print 'Loading reviews from pickle file:', PICKLE_FILE_PATH
    shuffled_labeled_reviews = pickle.load(open(PICKLE_FILE_PATH, 'rb'))

    X, y = SplitAndSparsifyXY(shuffled_labeled_reviews)
    if add_extra_features:
        AddPositiveNegativeWordFeatures(X, y)
    X_training, y_training, X_testing, y_testing = PartitionData(X, y)
    assert len(X_training) == len(y_training)
    assert len(X_testing) == len(y_testing)
    return X_training, y_training, X_testing, y_testing
