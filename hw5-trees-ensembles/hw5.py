# Machine Learning Homework 5
# 2015/04/29
# Alex Pine

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def parse_data_file(filename):
    X = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            label, input_a, input_b = line.split(',')
            X.append([float(input_a), float(input_b)])
            y.append(int(label))
    X = np.array(X)
    # Standardize X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    y = np.array(y)
    return X, y


def parse_data(use_zero_one_labels=False):
    X_train, y_train = parse_data_file('/Users/pinesol/mlcs/hw5-trees-ensembles/data/banana_train.csv')
    X_test, y_test = parse_data_file('/Users/pinesol/mlcs/hw5-trees-ensembles/data/banana_test.csv')
    return X_train, y_train, X_test, y_test


def tree_contour():
    '''Problem 3.3.1'''
    X_train, y_train, X_test, y_test = parse_data()
    # Set -1 labels to zero
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    MAX_TREE_DEPTH = 10

    NUM_PLOT_ROWS = 2
    NUM_PLOT_COLS = 5

    # Parameters
    n_labels = 2
    plot_colors = "br"
    plot_step = 0.02

    plt.figure(1, figsize = [24,14])

    for tree_depth in range(1, MAX_TREE_DEPTH+1):
        # Train
        clf = DecisionTreeClassifier(max_depth=tree_depth).fit(X_train, y_train)

        # Plot the decision boundary
        plt.subplot(NUM_PLOT_ROWS, NUM_PLOT_COLS, tree_depth)

        a_min, a_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        b_min, b_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        aa, bb = np.meshgrid(np.arange(a_min, a_max, plot_step),
                             np.arange(b_min, b_max, plot_step))

        Z = clf.predict(np.c_[aa.ravel(), bb.ravel()])
        Z = Z.reshape(aa.shape)
        cs = plt.contourf(aa, bb, Z, cmap=plt.cm.Paired)

        plt.xlabel('Feature A')
        plt.ylabel('Feature B')
        plt.axis("tight")

        # Plot the training points
        for label, color in zip(range(n_labels), plot_colors):
            indexes = np.where(y_test == label)
            plt.scatter(X_test[indexes, 0], X_test[indexes, 1], c=color, label=None,
                        cmap=plt.cm.Paired)

        plt.axis("tight")

    plt.suptitle("Decision surface of a decision tree using the Banana Dataset")
    plt.legend()
    plt.show()
    

def tree_error():
    '''Problem 3.3.3'''
    MAX_TREE_DEPTH = 10

    depths = range(1, MAX_TREE_DEPTH+1)
    train_error_values = []
    test_error_values = []

    X_train, y_train, X_test, y_test = parse_data()

    # Set -1 labels to zero
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    for tree_depth in depths:
        clf = DecisionTreeClassifier(max_depth=tree_depth).fit(X_train, y_train)

        y_train_hat = clf.predict(X_train)
        assert len(y_train) == len(y_train_hat)
        train_error = float(sum(y_train != y_train_hat)) / len(y_train)
        train_error_values.append(train_error)

        y_test_hat = clf.predict(X_test)
        assert len(y_test) == len(y_test_hat)
        test_error = float(sum(y_test != y_test_hat)) / len(y_test)
        test_error_values.append(test_error)

    plt.plot(depths, train_error_values, label='Train')
    plt.plot(depths, test_error_values, label='Test')
    plt.xlabel('Tree Depth')
    plt.ylabel('Error')
    plt.legend(loc='upper right')
    plt.show()


def tree_gridsearch():
    '''Problem 3.3.4'''
    MAX_TREE_DEPTH = 10
    MAX_SPLIT_SIZE = 10
    MAX_SAMPLES_LEAF = 10

    depths = range(1, MAX_TREE_DEPTH+1)
    min_splits = range(2, MAX_SPLIT_SIZE+1)
    min_samples_leaves = range(1, MAX_SAMPLES_LEAF+1)
    criterions = ['gini', 'entropy']

    smallest_test_error = 1.0
    best_depth = None
    best_min_samples_leaf = None
    best_criterion = None

    X_train, y_train, X_test, y_test = parse_data()
    # Set -1 labels to zero
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    for depth, min_samples, criterion in itertools.product(depths, min_samples_leaves, criterions):
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, min_samples_leaf=min_samples)
        clf = clf.fit(X_train, y_train)

        y_test_hat = clf.predict(X_test)
        assert len(y_test) == len(y_test_hat)
        test_error = float(sum(y_test != y_test_hat)) / len(y_test)
        if test_error < smallest_test_error:
            smallest_test_error = test_error
            best_depth = depth
            best_min_samples_leaf = min_samples
            best_criterion = criterion
    if best_depth:
        print 'Smallest Error:', smallest_test_error
        print 'Best Depth:', best_depth
        print 'Best Minimum Samples Leaf:', best_min_samples_leaf
        print 'Best Criterion:', criterion


def tree_adaboost(X_train, y_train, num_boosting_rounds):
    '''Problem 5.1'''
    MAX_TREE_DEPTH = 3
    data_weights = [1.0/len(X_train)]*len(X_train)
    data_weight_sum = 1.0
    classifiers = []
    classifier_weights = []

    for i in range(num_boosting_rounds):
        clf = DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH)
        clf = clf.fit(X_train, y_train, sample_weight=data_weights)
        classifiers.append(clf)
        y_hat = clf.predict(X_train)
        assert len(y_train) == len(y_hat)
        is_incorrect_list = [y_train_val != y_hat_val for y_train_val, y_hat_val in zip(y_train, y_hat)]
        error = sum([weight*is_incorrect 
                     for weight, is_incorrect in zip(data_weights, is_incorrect_list)]) / data_weight_sum
        classifier_weight = np.log((1-error)/error)
        classifier_weights.append(classifier_weight)
        data_weights = [data_weight*np.exp(classifier_weight*is_incorrect) 
                        for data_weight, is_incorrect in zip(data_weights, is_incorrect_list)]
        data_weight_sum = sum(data_weights)

    def adaboost_classify_func(X):
        return np.array([np.sign(sum([weight*clf.predict(X_row)[0]
                                      for weight, clf in zip(classifier_weights, classifiers)]))
                         for X_row in X])
    return adaboost_classify_func


def adaboost_error():
    '''Problem 5.3'''
    NUM_BOOSTING_ROUNDS = 10

    boosting_rounds = range(1, NUM_BOOSTING_ROUNDS+1)
    train_error_values = []
    test_error_values = []

    X_train, y_train, X_test, y_test = parse_data()
    for num_boosting_rounds in boosting_rounds:
        classify = tree_adaboost(X_train, y_train, num_boosting_rounds)
        y_train_hat = classify(X_train)
        assert len(y_train) == len(y_train_hat)
        train_error = float(sum(y_train != y_train_hat)) / len(y_train)
        train_error_values.append(train_error)

        y_test_hat = classify(X_test)
        assert len(y_test) == len(y_test_hat)
        test_error = float(sum(y_test != y_test_hat)) / len(y_test)
        test_error_values.append(test_error)

    plt.plot(boosting_rounds, train_error_values, label='Train')
    plt.plot(boosting_rounds, test_error_values, label='Test')
    plt.xlabel('Number of Boosting Rounds')
    plt.ylabel('Error')
    plt.legend(loc='upper right')
    plt.show()
   
    
if __name__ == '__main__':
    tree_gridsearch()
