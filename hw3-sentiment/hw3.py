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
    '''Question 3.2: The Pegasos Loss function.'''
    reg_term = lambda_reg * util.dotProduct(theta, theta) / 2.0
    margin = y_val * util.dotProduct(theta, x)
    loss_term = max(0.0, 1 - margin)
    return reg_term + loss_term
        

def PegasosSubgradientLoss(x, y_val, theta, lambda_reg):
    '''Question 3.2: The Subgradient of the Pegasos Loss function.'''
    margin = y_val * util.dotProduct(theta, x)
    subgrad = theta.copy()
    util.scale(subgrad, lambda_reg)
    if margin < 1:
        util.increment(subgrad, -y_val, x)
    return subgrad


def Pegasos(X, y, lambda_reg, max_epochs=1000, check_gradient=False):
    '''Question 4.2.
    Finds the sparse weight vector that minimizes the SVM loss function on X and y.
    '''
    print 'Running Pegasos with regularization parameter', lambda_reg
    loss_func = lambda x, y_val, theta: PegasosLoss(x, y_val, theta, lambda_reg)
    gradient_loss_func = lambda x, y_val, theta: PegasosSubgradientLoss(x, y_val, theta, lambda_reg)

    # Initialize theta to have zero for every word mentioned in any review
    theta = {key: 0.0 for x in X for key in x.keys()}
    t = 2 # NOTE: This normally starts at zero, but that causes a divide-by-zero error.
    weight_scalar = 1.0 

    for epoch in range(max_epochs):
#        print '--Epoch', epoch
        old_theta = theta.copy()
        for j, x in enumerate(X):
            t += 1
            eta = 1.0 / (t * lambda_reg)
            margin = y[j] * weight_scalar * util.dotProduct(theta, x)
            # NOTE that the gradient is not differentiable at 1.0, so we don't check it near there.
            if check_gradient and abs(margin-1.0) > 0.01:
                if SparseGradChecker(loss_func, gradient_loss_func, x, y[j], theta): 
                    print 'Computed gradient doesn\'t match approximations.'
                    sys.exit(1)
                grad = gradient_loss_func(x, y[j], theta)
                util.increment(theta, -eta, grad)
            else:
                weight_scalar *= 1.0 - 1.0/t
                if margin < 1:
                    util.increment(theta, eta * y[j]/weight_scalar, x)
        util.increment(old_theta, -1, theta)
        util.scale(old_theta, weight_scalar)
        total_change = math.sqrt(util.dotProduct(old_theta, old_theta))
#        print '----Change from previous theta:', total_change
        if total_change < 0.01:
            break
    util.scale(theta, weight_scalar)
    return theta

def PercentageWrong(X, y, theta):
    '''Question 4.3: The percentage incorrect when using theta to predict y from X.'''
    num_wrong = 0
    for i, x in enumerate(X):
        estimate_sign = np.sign(util.dotProduct(theta, x))
        if estimate_sign != y[i]:
            num_wrong += 1
    return 1.0 * num_wrong / len(y)
        
# best_lambda_reg 1e-05 smallest_loss 0.182
# Another run had best_lambda_reg 1e-05 smallest_loss 0.162
def FindBestRegularizationParameter(X_training, y_training, X_testing, y_testing,
                                    lower_bound_power, upper_bound_power):
    '''Question 4.4.
    Finds the regularization parameter that results in the smallest test loss by
    searching between the given powers of ten.
    That is, it finds the best regularization parameter between 10^lower_bound_power and 
    10^upper_bound_power.
    '''
    big_lambda_regs = [10.0**i for i in range(lower_bound_power, upper_bound_power)]
    assert len(big_lambda_regs) >= 2
    print 'Searching for best regularization parameters within:', big_lambda_regs
    best_lambda_reg_index = -1
    second_best_lambda_reg_index = -1
    smallest_loss = None
    second_smallest_loss = None
    for index, lambda_reg in enumerate(big_lambda_regs):
        theta = Pegasos(X_training, y_training, lambda_reg)
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
        theta = Pegasos(X_training, y_training, lambda_reg)
        loss = PercentageWrong(X_testing, y_testing, theta)
        print 'Regularization Parameter of', lambda_reg, 'produced loss of', loss
        if not smallest_loss or loss < smallest_loss:
            smallest_loss = loss
            best_lambda_reg = lambda_reg
    print 'best_lambda_reg', best_lambda_reg, 'smallest_loss', smallest_loss


def PlotScoresAgainstAccuracy(X_training, y_training, X_testing, y_testing, lambda_reg):
    '''Question 4.5.
    Divides the training set into buckets by score, and creates a bar chart showing the accuracy of
    each bucket.
    '''
    NUM_BUCKETS = 10

    theta = Pegasos(X_training, y_training, lambda_reg)
    # Calculate the score for each row in a list
    scores = [util.dotProduct(theta, x) for x in X_testing]
    
    low_score = min(scores)
    high_score = max(scores)

    # f(score) -> bucket
    score_to_bucket_func = lambda score: int(round((NUM_BUCKETS-1) * (score - low_score) / (high_score - low_score)))

    # Make a list of empty lists with NUM_BUCKETS elements
    # Each entry is a list of the indexes of X's rows that fall in the same score bucket.
    score_histogram = [[] for _ in range(NUM_BUCKETS)]
    for row_index, score in enumerate(scores):
        bucket = score_to_bucket_func(score)
        score_histogram[bucket].append(row_index)

    bucket_means = [0.0]*NUM_BUCKETS
    bucket_accuracy = [0.0]*NUM_BUCKETS

    for bucket, row_indices in enumerate(score_histogram):
    # calculate the percentage wrong loss for each bucket
    # make a scatter plot of these
        bucket_scores = [scores[row_index] for row_index in row_indices]
        bucket_score_mean = abs(np.mean(bucket_scores))
        bucket_means[bucket] = bucket_score_mean
        bucket_score_std = np.std(bucket_scores)
#        print 'Bucket', bucket, 'ranges from', min(bucket_scores), 'to', max(bucket_scores)
#        print 'Bucket', bucket, 'mean:', bucket_score_mean
#        print 'Bucket', bucket, 'stdev:', bucket_score_std
        X_bucket = [X_testing[row_index] for row_index in row_indices]
        y_bucket = [y_testing[row_index] for row_index in row_indices]
        bucket_accuracy[bucket] = 100*(1.0 - PercentageWrong(X_bucket, y_bucket, theta) )

    fig, ax = plt.subplots()
    ax.set_xlabel('Mean Score for Bucket')
    ax.set_ylabel('Percentage Correct')
    ax.set_title('Pegasos Sentiment Analysis: Score vs. Accuracy')
    width = 0.4
    positions = range(0, len(bucket_accuracy))
    rects1 = ax.bar(positions, bucket_accuracy, width, color='b', alpha=0.8)
    plt.xticks(rotation=-45)
    ax.set_xticks([pos + width for pos in positions])
    ax.set_xticklabels(["%0.1f" % mean for mean in bucket_means])
    plt.show()


def PrintReviewInfo(x_row, y_val, score, theta):
    '''Question 5.1.
    For a given review vector with an inc, prints if it's a false posi
    '''
    # Create map {score -> term_list}
    term_score_terms_map = collections.defaultdict(list)
    for term, count in x_row.iteritems():
        term_score = count * theta.get(term, 0.0)
        term_score_terms_map[term_score].append(term)
    MAX_TERMS = 500
    if np.sign(y_val) > 0 and np.sign(score) < 0:
        print 'False Negative Review. Score:', score
    elif np.sign(y_val) < 0 and np.sign(score) > 0:
        print 'False Positive Review. Score:', score
    else:
        print 'Correctly classified review. Score:', score
    print ''
    for term_score in sorted(term_score_terms_map.keys(), reverse=True, key=abs)[:MAX_TERMS]:
        terms = term_score_terms_map[term_score]
        for term in terms:
            print term, ', count:', x_row[term], ', weight:', theta.get(term, 0.0), ', term score:', term_score
    print '\n'

def ErrorAnalysis(X_training, y_training, X_testing, y_testing, lambda_reg):
    '''Question 5.1.
    Prints information about the top incorrect reviews, ordered by the magnitude of their score.
    '''
    theta = Pegasos(X_training, y_training, lambda_reg)
    scores = [util.dotProduct(theta, x) for x in X_testing]

    # (index, score) pairs, sorted by the score's absolute value in descending order.
    score_indexes = sorted(enumerate(scores), reverse=True, key=lambda index_score_pair: abs(index_score_pair[1]))

    num_incorrect_examples = 0
    MAX_NUM_WRONG_EXAMPLES = 10
    # Print out the information about all the incorrect examples, in order of largest score.
    for row_index, score in score_indexes:
        if num_incorrect_examples >= MAX_NUM_WRONG_EXAMPLES:
            break
        y_testing_val = y_testing[row_index]
        if np.sign(score) != np.sign(y_testing_val):
            x_testing_row = X_testing[row_index]
            PrintReviewInfo(x_testing_row, y_testing_val, score, theta)
            num_incorrect_examples += 1

    # TODO are the weights incorrect? Are significant words given a small weight? Maybe no one word appears that often.


def main():
    X_training, y_training, X_testing, y_testing = load.LoadData()
#    theta = Pegasos(X_training, y_training, lambda_reg)
#    FindBestRegularizationParameter(X_training, y_training, X_testing, y_testing, -8, -2)
    lambda_reg = 10**-5
#    PlotScoresAgainstAccuracy(X_training, y_training, X_testing, y_testing, lambda_reg)
    ErrorAnalysis(X_training, y_training, X_testing, y_testing, lambda_reg)    

    # TODO 5 error analysis
    # TODO 6 find a new feature that improves error

if __name__ == '__main__':
    main()



