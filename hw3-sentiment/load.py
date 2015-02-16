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
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

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
	
    pos_review = folder_list(pos_path, 1)
    neg_review = folder_list(neg_path, -1)
	
    review = pos_review + neg_review
    random.shuffle(review)
    return review


def SplitAndSparsifyXY(shuffled_labeled_reviews):
    '''
    Code for question 3.1.
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


def PartitionData(X, y):
    '''Code for question 2.1.'''
    NUM_TRAINING_ROWS = 500 # 1500 TODO TODO TODO
    X_training = X[:NUM_TRAINING_ROWS]
    y_training = y[:NUM_TRAINING_ROWS]
    X_testing = X[NUM_TRAINING_ROWS:]
    y_testing = y[NUM_TRAINING_ROWS:]
    assert len(X_training) + len(X_testing) == 2000
    assert len(y_training) + len(y_testing) == 2000
    return X_training, y_training, X_testing, y_testing
	

def LoadData():
    '''Code for question 2.1 and 3.1.'''
    if not os.path.isfile(PICKLE_FILE_PATH):
        print 'Pickle file not found, loading raw data...'
        shuffled_labeled_reviews = shuffle_data()
        pickle.dump(shuffled_labeled_reviews, open(PICKLE_FILE_PATH, 'wb'))
    else:
        print 'Loading reviews from pickle file:', PICKLE_FILE_PATH
    shuffled_labeled_reviews = pickle.load(open(PICKLE_FILE_PATH, 'rb'))

    X, y = SplitAndSparsifyXY(shuffled_labeled_reviews)
    X_training, y_training, X_testing, y_testing = PartitionData(X, y)
    assert len(X_training) == len(y_training)
    assert len(X_testing) == len(y_testing)
    return X_training, y_training, X_testing, y_testing
