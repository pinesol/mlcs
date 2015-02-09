# Alex Pine
# 2015/02/07
# Homework 2

import logging
import math
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import pandas as pd
import scipy.optimize
from sklearn.cross_validation import train_test_split
import sklearn.linear_model

NUM_ROWS = 150
NUM_FEATURES = 75
ZERO_THRESHOLD = 10**-1


def RandomDesignMatrix():
    '''
    1.1: m = 150 examples, each of dimension d = 75
    Construct a random design matrix X using numpy.random.rand() function.
    '''
    return np.random.rand(NUM_ROWS, NUM_FEATURES)


def RandomSparseThetaVector():
    '''
    1.1.2: Construct a true weight vector theta in R^(dx1) as follows: set the
    first 10 components of theta to 10 or -10 arbitrarilyi, and all the other
    components to zero.
    '''
    theta = np.zeros((NUM_FEATURES, 1))
    for row_index in range(10):
        theta[row_index] = random.choice([-10, 10])
    return theta


def ComputeNoiseyResponseVector(design_matrix, theta_vector):
    '''1.1.3: Response vector for design*theta + noise.'''
    standard_deviation = 0.1
    noise_vector = standard_deviation * np.random.randn(design_matrix.shape[0], 1)
    return np.dot(design_matrix, theta_vector) + noise_vector


def PartitionDataset(design_matrix, response_vector):
    '''
    1.1.4: Split the dataset by taking the first 80 points for training, the
    next 20 points for validation, and the last 50 points for testing.
    '''
    training_matrix = design_matrix[:80]
    training_response_vector = response_vector[:80]
    validation_matrix = design_matrix[80:100]
    validation_response_vector = response_vector[80:100]
    testing_matrix = design_matrix[100:]
    testing_vector = response_vector[100:]
    return (training_matrix, training_response_vector, 
            validation_matrix, validation_response_vector, 
            testing_matrix, testing_vector)
    

def ridge(X, y, lambda_reg):
  def ridge_obj(theta):
    return ((linalg.norm(np.dot(X,theta) - y))**2)/(2*X.shape[0]) + lambda_reg*(linalg.norm(theta))**2
  return ridge_obj


def compute_loss(X, y, theta):
  return ((linalg.norm(np.dot(X,theta) - y))**2)/(2*X.shape[0])


# TODO re-run this now taht we've relaxed the zero threshold
def RidgeRegressionExperiments(lambda_regs):
    #    1.2 Experiments with Ridge Regression
    design_matrix = RandomDesignMatrix()
    true_theta_vector = RandomSparseThetaVector()
    true_theta_zero_indices = [i for i in range(len(true_theta_vector)) if true_theta_vector[i] == 0]
    response_vector = ComputeNoiseyResponseVector(design_matrix, true_theta_vector)

    X_training, y_training, X_validation, y_validation, X_testing, y_testing = PartitionDataset(design_matrix, response_vector)

    # find lambda that minimizes loss on the square loss on the training set
    random_theta_vector = np.random.rand(true_theta_vector.shape[0], 1)

    smallest_validation_loss = None
    best_lambda_reg = None
    best_theta = None

    smallest_test_validation_loss = None
    best_test_lambda_reg = None
    best_test_theta = None

    for lambda_reg in lambda_regs:
        # Compute ridge regression using explicit function 'ridge', and minimize the loss with scipy.optimize.minimize
        opt_result = scipy.optimize.minimize(ridge(X_training, y_training, lambda_reg), random_theta_vector)
        theta = opt_result.x
        # Find the save the lambda_reg and theta with the smallest validation loss.
        validation_loss = compute_loss(X_validation, y_validation, theta)
        if smallest_validation_loss is None or validation_loss < smallest_validation_loss:
            smallest_validation_loss = validation_loss
            best_lambda_reg = lambda_reg
            best_theta = theta
        
        # Use sklearn's ridge regression classifier to test your manual one.
        test_classifier = sklearn.linear_model.Ridge(alpha=lambda_reg, fit_intercept=False).fit(X_training, y_training) # TODO fit_intercept
        test_theta = test_classifier.coef_.reshape(test_classifier.coef_.shape[1]) # Reshape the test theta to be a column vector
        assert theta.shape == test_theta.shape, 'theta_shape: %s, test_theta_shape: %s'  % (theta.shape, test_theta.shape)
        # Find the save the lambda_reg and theta with the smallest validation loss as determined by the sklearn classifier.
        test_validation_loss = compute_loss(X_validation, y_validation, test_theta)
        if smallest_test_validation_loss is None or test_validation_loss < smallest_test_validation_loss:
            smallest_test_validation_loss = test_validation_loss
            best_test_lambda_reg = lambda_reg
            best_test_theta = test_theta

    print 'my model'
    print smallest_validation_loss, best_lambda_reg
    print 'best_theta', best_theta
    num_exactly_zero = sum([theta_i == 0.0 for theta_i in best_theta])
    print 'num_exactly_zero', num_exactly_zero

    num_near_zero = sum([abs(theta_i - 0.0) < ZERO_THRESHOLD for theta_i in best_theta])
    print 'num_near_zero', num_near_zero

    print 'sklearn\'s model'
    print smallest_test_validation_loss, best_test_lambda_reg
    print 'best_test_theta', best_test_theta
    num_test_exactly_zero = sum([theta_i == 0.0 for theta_i in best_test_theta])
    print 'num_test_exactly_zero', num_test_exactly_zero

    num_test_near_zero = sum([abs(theta_i - 0.0) < ZERO_THRESHOLD for theta_i in best_test_theta])
    print 'num_test_near_zero', num_test_near_zero

    # test if the important metrics match
    lambda_reg_matches = best_lambda_reg == best_test_lambda_reg
    num_exactly_zero_matches = num_exactly_zero == num_test_exactly_zero
    num_near_zero_matches = num_near_zero == num_test_near_zero
    print 'lambda_reg_matches: %s, num_exactly_zero_matches: %s, num_near_zero_matches: %s' % (lambda_reg_matches, num_exactly_zero_matches, num_near_zero_matches)


def FitShootingLasso(X, y, lambda_reg, starting_theta):
    '''2.1.1'''
    # Function Constants
    converged_distance = 10**-3 # totally arbitrary convergence definition
    max_iterations = 1000 # Aribrary limit to ensure this doesn't run forever
    num_rows = X.shape[0]
    num_features = X.shape[1]

    # Iteration variables
    theta = starting_theta
    theta_distance = None
    iteration = 0

    while (theta_distance is None or theta_distance > converged_distance) and iteration < max_iterations:
        old_theta = np.copy(theta)
        for j in range(num_features):
            a_j = 0.0
            c_j = 0.0
            for i in range(num_rows):
                a_j += X[i][j]**2
                c_j += X[i][j] * (y[i] - np.dot(theta.T, X[i]) + theta[j] * X[i][j]) # TODO Should I use theta or old_theta here?
            a_j *= 2
            c_j *= 2
            a = c_j/a_j
            theta[j] = np.sign(a) * np.max(abs(a) - lambda_reg/a_j, 0.0)
        theta_distance = linalg.norm(theta - old_theta)
        iteration += 1
#    print 'theta distance', theta_distance, 'converged distance', converged_distance
#    print 'iterations until convergence', iteration
    if iteration == max_iterations:
        print 'Maximum iterations reached while calculating shooting lasso for lambda %f, stopping optimization.' % (lambda_reg)
    return theta


def printThetaComparisionInfo(true_theta, best_theta):
    # Determine which values of theta it calculated correctly
    num_true_non_zeros = 0
    num_true_zeros = 0
    for i in range(len(true_theta)):
        true_theta_i = true_theta[i]
        best_theta_i = best_theta[i]
        if abs(true_theta_i - best_theta_i) < ZERO_THRESHOLD:
            if i < 10: num_true_non_zeros += 1
            else: num_true_zeros += 1
    print 'Number of true zeros:', num_true_zeros, 'out of 65.'
    print 'Number of true non-zeros (10 or -10):', num_true_non_zeros, 'out of 10'

        
def PlotValidationLossAgainstLambdas(X, y, true_theta, lambda_regs):
    X_training, y_training, X_validation, y_validation, X_testing, y_testing = PartitionDataset(X, y)

    # These are used to calculate the starting theta value
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)

    validation_losses = []*len(lambda_regs)
    smallest_validation_loss = None
    best_lambda_reg = None
    best_theta = None

    for lambda_reg in lambda_regs:
        starting_theta = np.dot(linalg.inv(XTX + np.dot(lambda_reg, np.identity(X.shape[1]))), XTy)
        theta = FitShootingLasso(X_training, y_training, lambda_reg, starting_theta)
        validation_loss = compute_loss(X_validation, y_validation, theta)

        validation_losses.append(validation_loss)

        if smallest_validation_loss is None or validation_loss < smallest_validation_loss:
            smallest_validation_loss = validation_loss
            best_lambda_reg = lambda_reg
            best_theta = theta

    testing_loss = compute_loss(X_testing, y_testing, best_theta)
    print 'best lambda: %f, testing loss: %f' % (best_lambda_reg, testing_loss)

    printThetaComparisionInfo(true_theta, best_theta)

    fig, ax = plt.subplots()
    ax.set_xlabel('log10 of Regularization Constant (Lambda)')
    ax.set_ylabel('log10 of Validation Loss')
    ax.set_title('Shooting Algorithm: Lambda vs. Loss')    
    lambda_powers = [math.log(lambda_reg, 10) for lambda_reg in lambda_regs]
    plt.xticks(lambda_powers)
    plt.plot(lambda_powers,
             [math.log(loss, 10) for loss in validation_losses])
    plt.show()


def ShootingHomotopy(X, y, true_theta):
    X_training, y_training, X_validation, y_validation, X_testing, y_testing = PartitionDataset(X, y)

    mean_y = np.mean(y_training, axis=0)
    lambda_reg_inf = linalg.norm(np.dot(X_training.T, y_training - mean_y), np.inf)
    print 'lambda_reg_inf', lambda_reg_inf

    # TODO we're supposed to keep going until the validation (test?) error gets to zero, but I'm doing 6 iterations, tops.
    lambda_regs = [lambda_reg_inf / 10.0**i for i in range(0, 6)]
    theta = np.zeros(true_theta.shape)

    stopping_validation_loss_diff = 10**-3 # TODO arbitrary
    smallest_validation_loss = None
    best_lambda_reg = None
    best_theta = None

    iteration = 0
    validation_loss = None

    # TODO check that this stopping criteria is correct
    # TODO figure out what the two things we're supposed to compare in terms of running speed
    # TODO since theta is updated every iteration, maybe it doesn't make sense to choose teh best. maybe we should just choose the last one.
    # TODO at the moment, this doesn't work at all
    while (iteration < len(lambda_regs) and
           (smallest_validation_loss is None 
            or abs(validation_loss - smallest_validation_loss) > stopping_validation_loss_diff)):
        lambda_reg = lambda_regs[iteration]
        theta = FitShootingLasso(X_training, y_training, lambda_reg, theta)
        validation_loss = compute_loss(X_validation, y_validation, theta)

        if smallest_validation_loss is None or validation_loss < smallest_validation_loss:
            smallest_validation_loss = validation_loss
            best_lambda_reg = lambda_reg
            best_theta = theta

        iteration += 1

    testing_loss = compute_loss(X_testing, y_testing, best_theta)
    print 'best lambda: %f, testing loss: %f' % (best_lambda_reg, testing_loss)

    printThetaComparisionInfo(true_theta, best_theta)
    

if __name__ == '__main__':
#    RidgeRegressionExperiments()
    X = RandomDesignMatrix()
    true_theta = RandomSparseThetaVector()
    y = ComputeNoiseyResponseVector(X, true_theta)
#    lambda_regs = [10**i for i in range(-4, 2)]
#    PlotValidationLossAgainstLambdas(X, y, true_theta, lambda_regs)
    ShootingHomotopy(X, y, true_theta)
