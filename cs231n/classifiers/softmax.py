import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[0]
    for idx in range(N):
        example = X[idx]
        scores = example.dot(W)
        scores -= np.max(scores)
        s_exp = np.exp(scores)
        sum_s_exp = np.sum(s_exp)
        loss += -np.log(s_exp[y[idx]] / sum_s_exp)
        for c in range(dW.shape[1]):  # for each column
            ex = s_exp[c]
            if c == y[idx]:
                dW.T[c] += (ex / sum_s_exp - 1.0) * example
            else:
                dW.T[c] += ex / sum_s_exp * example
    loss /= N
    loss += reg * np.sum(W * W)
    dW /= N
    dW += 2 * W * reg
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[0]
    scores = X.dot(W)  # N,C
    max_score = np.max(scores, axis=1)
    scores -= max_score[:, np.newaxis]
    scores_exp = np.exp(scores)  # N,C
    sum_scores_exp = np.sum(scores_exp, axis=1)  # N
    scores_exp_correct_label = scores_exp[range(scores.shape[0]), y]
    loss = -np.log(scores_exp_correct_label / sum_scores_exp)
    loss = np.sum(loss)
    loss /= N
    loss += reg * np.sum(W * W)

    # gradient
    a = scores_exp / sum_scores_exp[:, np.newaxis]
    a[range(N), y] -= 1
    dW = X.T.dot(a)  # D,N dot N,C
    dW /= N
    dW += 2 * W * reg
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
