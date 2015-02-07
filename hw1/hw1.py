# Alex Pine
# 2015/02/01
# Homework 1

import logging
import math
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import pandas as pd
from sklearn.cross_validation import train_test_split

DATA_FILE = '/Users/pinesol/machine-learning/hw1-sgd/hw1-data.csv'

### Assignment Owner: Hao Xu

#######################################
####Q2.1: Normalization

def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.
    
    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    # train dimensions: [n x m]
    # test dimensions:  [k x m]
    # transformation: (value - column_minimum) / (column_maximum - column_minimum)

    train_min = train.min(axis=0) # [1 x m]
    train_max_minus_min = train.max(axis=0) - train_min # [1 x m]
    normalized_train = 1.0 * (train - train_min) / train_max_minus_min # ([n x m] - [1 x m]) / [1 x m] = [n x m]
    normalized_test = 1.0 * (test - train_min) / train_max_minus_min # ([k x m] - [1 x m]) / [1 x m] = [k x m]
    for row in normalized_train:
        for value in row:
            assert value >= 0.0 and value <= 1.0
    return normalized_train, normalized_test


########################################
####Q2.2a: The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)
    
    Returns:
        loss - the square loss, scalar
    """
    num_instances = X.shape[0]
    return linalg.norm(np.dot(X, theta) - y)**2 / (2*num_instances)


########################################
###Q2.2b: compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    num_instances = X.shape[0]
    return np.dot(np.dot(X,theta) - y, X) / num_instances


###########################################
###Q2.3a: Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4): 
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions: 
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1) 

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by: 
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate

    for i in range(num_features):
        theta_plus = np.copy(theta)
        theta_plus[i] += epsilon
        theta_minus = np.copy(theta)
        theta_minus[i] -= epsilon
        approx_grad[i] = (compute_square_loss(X, y, theta_plus) - compute_square_loss(X, y, theta_minus)) / (2*epsilon)
    distance = linalg.norm(true_gradient - approx_grad)
    if distance > tolerance:
        print 'Gradient doesn\'t match approximation. Distance: %s, Tolerance: %s. Grad: %s, Approx: %s' % (distance, tolerance, true_gradient, approx_grad)
        return False
    return True


###Q2.3b: Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        objective_func = the objective function, takes X, Y, and theta as input, returns a scalar as output.
        gradient_func = the function that computes the gradient of objective function. Takes X, Y, and
          theta as input, returns an array with the same dimensions theta as output.
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Returns:
        A boolean value indicate whether the gradient is correct or not
    """
    true_gradient = gradient_func(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate

    for i in range(num_features):
        theta_plus = np.copy(theta)
        theta_plus[i] += epsilon
        theta_minus = np.copy(theta)
        theta_minus[i] -= epsilon
        approx_grad[i] = (objective_func(X, y, theta_plus) - objective_func(X, y, theta_minus)) / (2*epsilon)
    distance = linalg.norm(true_gradient - approx_grad)
    if distance > tolerance:
        print 'Gradient doesn\'t match approximation. Distance: %s, Tolerance: %s. Grad: %s, Approx: %s' % (distance, tolerance, true_gradient, approx_grad)
        return False
    return True


####################################
####Q2.4a: Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run 
        check_gradient - a boolean value indicating whether checking the gradient when updating
        
    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1) 
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.ones(num_features) #initialize theta
    for i in range(num_iter+1):
        # Record the current value of theta and the loss it causes.
        theta_hist[i] = theta
        loss_hist[i] = compute_square_loss(X, y, theta)
        # Compute the gradient and check its correctness.
        grad = compute_square_loss_gradient(X, y, theta)
        if check_gradient and not grad_checker(X, y, theta):
            return None, None
        # Update theta
        theta = theta - alpha * grad / linalg.norm(grad) 
    return theta_hist, loss_hist


def plotBatchGradientDescentConvergence(alphas, num_iterations):
    df = pd.read_csv(DATA_FILE, delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1)))) #Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) #Add bias term

    loss_histories = [batch_grad_descent(X_train, y_train, alpha=alpha, num_iter=num_iterations, check_gradient=True)[1]
                      for alpha in alphas]

    fig, ax = plt.subplots()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Batch Grad Descent: Alpha vs. Loss')    
    legends = []

    for alpha, loss_hist in zip(alphas, loss_histories):
        plt.plot(loss_hist)
        legends.append('%0.3f' % (alpha))
    plt.legend(legends, title='Alpha Value', loc='upper right')
    plt.show()


def compute_regularized_square_loss(X, y, theta, lambda_reg):
    '''This uses the above value for lambda_reg so this can be passed to generic_gradient_checker.'''
    num_instances = X.shape[0]
    reg_term = lambda_reg * np.dot(theta, theta)
    return (linalg.norm(np.dot(X, theta) - y)**2) / (2*num_instances) + reg_term


###Q2.5a: Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    num_instances = X.shape[0]
    return (np.dot(np.dot(X, theta) - y, X) / num_instances) + (2 * lambda_reg * theta)

###################################################
###Q2.5b: Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run 
        
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features) 
        loss_hist - the history of regularized loss value, 1D numpy array
    """
    (num_instances, num_features) = X.shape
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist

#    ts = []
    for i in range(0, num_iter+1):
        theta_hist[i] = theta
        loss_hist[i] = compute_square_loss(X, y, theta)

        if not generic_gradient_checker(X, y, theta,
                                       lambda x_c,y_c,theta_c: compute_regularized_square_loss(x_c, y_c, theta_c, lambda_reg),
                                       lambda x_c,y_c,theta_c: compute_regularized_square_loss_gradient(x_c, y_c, theta_c, lambda_reg)):
           return None, None
#        before_time = time.time()
        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta = theta - alpha * grad / linalg.norm(grad)
#        ts.append(time.time() - before_time)
#    print 'average time to compute gradient step', 1000 * sum(ts) / len(ts), 'ms'
    return theta_hist, loss_hist


def plotRidgeRegressionLossAgainstLambda(alpha, num_iterations, lambda_powers, Bs):
    df = pd.read_csv(DATA_FILE, delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)

    lambda_regs = [10**power for power in lambda_powers]
    # Bias -> (training loss for each lambda, validation loss for each lambda_reg)
    bias_map = {}
    for bias in Bs:
        X_train = np.hstack((X_train, bias * np.ones((X_train.shape[0], 1)))) #Add bias term
        X_test = np.hstack((X_test, bias * np.ones((X_test.shape[0], 1)))) #Add bias term

        training_losses = []
        validation_losses = []
        for lambda_reg in lambda_regs:
            theta_hist, loss_hist = regularized_grad_descent(X_train, y_train, alpha, lambda_reg,
                                                             num_iter=num_iterations)
            # calculate training loss
            training_losses.append(loss_hist[-1])
            # calculate test (validation) loss
            validation_loss = compute_square_loss(X_test, y_test, theta_hist[-1])
            validation_losses.append(validation_loss)
        bias_map[bias] = (training_losses, validation_losses)

    fig, ax = plt.subplots()
    ax.set_xlabel('log_10(lambda)') 
    ax.set_ylabel('Loss')
    ax.set_title('Ridge Regression: Lambda vs. Loss')
    legends = []

    for bias, loss_tuple in bias_map.iteritems():
        training_losses, validation_losses = loss_tuple
        plt.plot(lambda_powers, training_losses, '--')
        plt.xticks(lambda_powers)
        legends.append('Training Loss: B = %d' % (bias))
        plt.plot(lambda_powers, validation_losses)
        plt.xticks(lambda_powers)
        legends.append('Validation Loss: B = %d' % (bias))
    plt.legend(legends, loc='upper left')
    plt.show()
    
    
#############################################
##Q2.5c: Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss

#############################################
###Q2.6a: Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set
    
    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features) 
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta    
    theta_hist = np.zeros((num_iter, num_instances, num_features))
    loss_hist = np.zeros((num_iter, num_instances))

    alpha_update_fn = None
    if not isinstance(alpha, float):
        if alpha == "1/sqrt(t)":
            alpha_update_fn = lambda t: 1.0 / math.sqrt(t)
        elif alpha == "1/t":
            alpha_update_fn = lambda t: 1.0 / t
        else: 
            print "Invalid string value for alpha"
            return None

    shuffled_row_indices = range(num_instances)
    random.shuffle(shuffled_row_indices)

#    ts = []
    for epoch in range(num_iter):
#        before_time = time.time()
        for i, row_index in enumerate(shuffled_row_indices):
            if alpha_update_fn is not None:
                t = epoch*len(shuffled_row_indices) + i + 100 # 100 is arbitrary, helps it converge more quickly
                alpha = alpha_update_fn(t)
            theta_hist[epoch][i] = theta
            loss_hist[epoch][i] = compute_regularized_square_loss(X, y, theta, lambda_reg) 

            X_row = X[row_index]
            y_row = y[row_index]

            reg_term = 2 * lambda_reg * theta
            grad = (np.dot(X_row, theta) - y_row) * X_row + reg_term
            theta = theta - alpha * (grad)
#        ts.append(time.time() - before_time)
#    print 'average time to compute stochastic gradient step', 1000 * sum(ts) / len(ts), 'ms'

    return theta_hist, loss_hist    


################################################
###Q2.6b Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value)

def main(bias, lambda_reg, num_iter, alphas):
    #Loading the dataset
    print('loading the dataset')
    
    df = pd.read_csv(DATA_FILE, delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, bias * np.ones((X_train.shape[0], 1)))) #Add bias term
    X_test = np.hstack((X_test, bias * np.ones((X_test.shape[0], 1)))) #Add bias term

    losses = []
    for alpha in alphas:
        print 'Training using stochastic gradient descent with alpha', alpha
        theta_hist, loss_hist = stochastic_grad_descent(X_train, y_train, alpha=alpha, lambda_reg=lambda_reg, num_iter=num_iter)
        losses.append([math.log(loss_hist[epoch][-1], 10) for epoch in range(num_iter)]) # Record the last theta after each epoch

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('log10(Regularized Loss)')
    ax.set_title('Stochastic Gradient Descent: Epoch vs. Loss')
    legends = []

    for alpha, loss_hist in zip(alphas, losses):
        plt.plot(loss_hist)
        legends.append('Alpha %s' % (str(alpha)))
    plt.legend(legends, loc='upper right')
    plt.show()        
    

if __name__ == "__main__":
    main(bias=1, lambda_reg=0.001, num_iter=1000, alphas=[.02, "1/t", "1/sqrt(t)"])
#    plotBatchGradientDescentConvergence([.1, .05, .02], 500)
#    plotRidgeRegressionLossAgainstLambda(lambda_powers=range(-4, 1),  Bs=[1, 2, 4])
