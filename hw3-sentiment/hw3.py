# Homework 3 - hw3.py
# Alex Pine
# 2015/02/15

import load
import util

import matplotlib.pyplot as plt
import numpy as np

import collections
import math
import sys


def SparseGradChecker(loss_func, gradient_loss_func, x, y_val, theta, epsilon=0.01, tolerance=1e-4): 
    """Question 3.2: Implement Generic Gradient Checker for Sparse Matrices.

    Check that the function gradient_loss_func returns the correct gradient for 
    the given x, y_val, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions: 
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1) 

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by: 
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    gradient_loss_func(x, y_val, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        loss_func - A function that computes the loss for (x, y_val, theta).
        gradient_loss_func - A function that computes gradient for (x, y_val, theta).
        x - A single row in the design matrix, represented by a dict/Counter object. (key length = num_features)
        y_val - the label for the corresponding x_row (-1 or 1)
        theta - the parameter vector, dict/Counter object. (key length = num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = gradient_loss_func(x, y_val, theta)
    approx_grad = dict.fromkeys(theta.keys(), 0.0)

    for key in theta.iterkeys():
        # Compute the approximate directional derivative in the chosen direction
        # Avoid copying since it's so slow.
        theta_key_original = theta[key]
        theta[key] += epsilon
        plus_loss = loss_func(x, y_val, theta)
        theta[key] = theta_key_original - epsilon
        minus_loss = loss_func(x, y_val, theta)
        theta[key] = theta_key_original # restore theta
        approx_grad[key] = (plus_loss - minus_loss) / (2*epsilon)
    util.increment(approx_grad, -1, true_gradient) # approx_grad - true_gradient
    error = math.sqrt(util.dotProduct(approx_grad, approx_grad)) # np.linalg.norm(approx_grad - true_gradient)
    if error > tolerance:
        print 'gradient doesn\'t match approximation. Error:', error
    return (error < tolerance)


def PegasosLoss(x, y_val, theta, lambda_reg):
    reg_term = lambda_reg * util.dotProduct(theta, theta) / 2.0
    margin = y_val * util.dotProduct(theta, x)
    loss_term = max(0.0, 1 - margin)
    return reg_term + loss_term
        

def PegasosSubgradientLoss(x, y_val, theta, lambda_reg):
    margin = y_val * util.dotProduct(theta, x)
    subgrad = theta.copy() # TODO untested
    util.scale(subgrad, lambda_reg)
    if margin < 1:
        util.increment(subgrad, -y_val, x)
    return subgrad


def Pegasos(X, y, lambda_reg, max_epochs=50, check_gradient=False, fast_method=False):
    '''Question 4.2.
    
    TODO write comments
    '''
    print 'Pegasos with regularization parameter', lambda_reg
    loss_func = lambda x, y_val, theta: PegasosLoss(x, y_val, theta, lambda_reg)
    gradient_loss_func = lambda x, y_val, theta: PegasosSubgradientLoss(x, y_val, theta, lambda_reg)

    # Initialize theta to have zero for every word mentioned in any review
    theta = {key: 0.0 for x in X for key in x.keys()}
    t = 0

    for epoch in range(max_epochs):
        print '--Epoch', epoch
        # TODO copy the previous version of theta, then take the difference at the end to see if you can stop iterating?
        old_theta = theta.copy()
        for j, x in enumerate(X):
            t += 1
            eta = 1.0 / (t * lambda_reg)
            if fast_method:
                # NOTE: This way is closer to the implemention described in the assignment. 
                # It's also faster than the version that computes the gradient explicitly, 
                # since it doesn't have to copy theta. However, you can't check the gradient
                # this way.
                margin = y[j] * util.dotProduct(theta, x)
                util.scale(theta, 1 - eta * lambda_reg)
                if margin < 1:
                    util.increment(theta, eta * y[j], x)
            else:
                if check_gradient and not SparseGradChecker(loss_func, gradient_loss_func, x, y[j], theta): 
                    print 'Computed gradient doesn\'t match approximations.'
                    sys.exit(1)
                grad = gradient_loss_func(x, y[j], theta)
                util.increment(theta, -eta, grad)
        util.increment(old_theta, -1, theta)
        total_change = math.sqrt(util.dotProduct(old_theta, old_theta))
        print '----Change from previous theta:', total_change
        if total_change < 1.0: # TODO this seems large, but I can't wait all day
            break
    return theta


# TODO is this right?
def PercentageWrong(X, y, theta):
    '''Question 4.3.'''
    num_wrong = 0
    for i, x in enumerate(X):
        estimate_sign = np.sign(util.dotProduct(theta, x))
        if estimate_sign != y[i]:
            num_wrong += 1
    return 1.0 * num_wrong / len(y)
        

def FindBestRegularizationParameter(X_training, y_training, X_testing, y_testing):
    '''Question 4.4.
    TODO write comments
    '''
    # TODO they use a tiny lambda (10^-5) in the paper, but when I use one, it takes many epochs
    big_lambda_regs = [10.0**i for i in range(-2, 2)] # NOTE this list must have at least two elements
    assert len(big_lambda_regs) >= 2
    print 'Searching for best regularization parameters within:', big_lambda_regs
    best_lambda_reg_index = -1
    second_best_lambda_reg_index = -1
    smallest_loss = None
    second_smallest_loss = None
    for index, lambda_reg in enumerate(big_lambda_regs):
        theta = Pegasos(X_training, y_training, lambda_reg, fast_method=True)
        loss = PercentageWrong(X_testing, y_testing, theta)
        print 'Regularization Parameter of', lambda_reg, 'produced loss of', loss
        if not smallest_loss or loss < smallest_loss:
            second_smallest_loss = smallest_loss
            smallest_loss = loss
            second_best_lambda_reg_index = best_lambda_reg_index
            best_lambda_reg_index = index
        elif not second_smallest_loss or loss < second_smallest_loss:
            second_smallest_loss = loss
            second_best_lambda_reg_index = index

    lower_best_lambda_reg = big_lambda_regs[min(second_best_lambda_reg_index, best_lambda_reg_index)]
    upper_best_lambda_reg = big_lambda_regs[max(second_best_lambda_reg_index, best_lambda_reg_index)]
    print 'Best regularization constant is between', lower_best_lambda_reg, 'and', upper_best_lambda_reg

    NUM_STEPS = len(big_lambda_regs)
    step_size = (upper_best_lambda_reg - lower_best_lambda_reg) / NUM_STEPS
    small_lambda_regs = [lower_best_lambda_reg + step_size*step for step in range(NUM_STEPS-1)] + [upper_best_lambda_reg]
    print 'Now searching for best regularization parameter within:', small_lambda_regs

    best_lambda_reg = None
    smallest_loss = None

    for lambda_reg in small_lambda_regs:
        theta = Pegasos(X_training, y_training, lambda_reg, fast_method=True)
        loss = PercentageWrong(X_testing, y_testing, theta)
        print 'Regularization Parameter of', lambda_reg, 'produced loss of', loss
        if not smallest_loss or loss < smallest_loss:
            smallest_loss = loss
            best_lambda_reg = lambda_reg
    print 'best_lambda_reg', best_lambda_reg, 'smallest_loss', smallest_loss


def PlotScoresAgainstAccuracy(X_training, y_training, X_testing, y_testing, lambda_reg):
    NUM_BUCKETS = 10

    theta = Pegasos(X_training, y_training, lambda_reg, fast_method=True)
    # Calculate the score for each row in a list
    scores = [util.dotProduct(theta, x) for x in X_testing]

    low_score = min(scores)
    high_score = max(scores)

    # f(score) -> bucket
    score_to_bucket_func = lambda score: int(round((NUM_BUCKETS-1) * (score - low_score) / (high_score - low_score)))
    # f(bucket) -> score # TODO not sure about this...
    bucket_to_score_func = lambda bucket: int(round(bucket * (high_score - low_score) / (NUM_BUCKETS-1) + low_score))

    # Make a list of empty lists with NUM_BUCKETS elements
    # Each entry is a list of the indexes of X's rows that fall in the same score bucket.
    score_histogram = [[]]*NUM_BUCKETS
     
    for row_index, score in enumerate(scores):
        score_histogram[score_to_bucket_func(score)].append(row_index)

    bucket_means = [0.0]*NUM_BUCKETS
    bucket_losses = [0.0]*NUM_BUCKETS

    for bucket, row_indices in enumerate(score_histogram):
    # calculate the percentage wrong loss for each bucket
    # make a scatter plot of these
        bucket_scores = [scores[row_index] for row_index in row_indices]
        bucket_score_mean = np.mean(bucket_scores)
        bucket_means[bucket] = bucket_score_mean
        bucket_score_std = np.std(bucket_scores)
        print 'Bucket', bucket, 'ranges from', min(bucket_scores), 'to', max(bucket_scores)
        print 'Bucket', bucket, 'mean:', bucket_score_mean
        print 'Bucket', bucket, 'stdev:', bucket_score_std
        # TODO
#        assert score_to_bucket_func(min(bucket_scores)) == score_to_bucket_func(max(bucket_scores))    
        X_bucket = [X_testing[row_index] for row_index in row_indices]
        y_bucket = [y_testing[row_index] for row_index in row_indices]
        loss = PercentageWrong(X_bucket, y_bucket, theta)
        bucket_losses[bucket] = loss

    fig, ax = plt.subplots()
    ax.set_xlabel('Mean Score for Bucket') # TODO maybe make the x label be the mean score for the bucket?
    ax.set_ylabel('Percentage Wrong')
    ax.set_title('Pegasos Sentiment Analysis: Score vs. Loss')    
    plt.xticks(bucket_means)
    plt.plot(bucket_means, bucket_losses)
    plt.show()


def main():
    X_training, y_training, X_testing, y_testing = load.LoadData()
#    print 'running pegasos'
#    Pegasos(X_training, y_training, 0.01, max_epochs=30, check_gradient=True)
#    FindBestRegularizationParameter(X_training, y_training, X_testing, y_testing)
    PlotScoresAgainstAccuracy(X_training, y_training, X_testing, y_testing, 0.01) # TODO even smaller lambdas are better, but they take forever to converge

    # TODO write up subgradient answer for question 4.1
    # TODO 4.4 Find the real best lambda_reg for question 4.4. The paper suggests it's 10^-.5
    #  - Use a smaller convergence threshold in Pegasos, and use the REAL data size split.
    # TODO 4.5 Fix: PlotScoresAgainstAccuracy.
    #  - It doesn't work! Every bucket has the whole range of scores!
    #  - use the best lambda and real data split
    #  - use more/fewer buckets
    #  - write up answer
    # TODO 5 error analysis
    # TODO 6 find a new feature that improves error

if __name__ == '__main__':
    main()



