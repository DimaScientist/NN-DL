from __future__ import annotations

import numpy as np


def softmax_loss_naive(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> :
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW_each = np.zeros_like(dW)
    num_train, _ = X.shape
    num_classes = W.shape[1]
    func = X.dot(W)

    func_max = np.reshape(np.max(func, axis=1), (num_train, 1))
    probability = np.exp(func - func_max) / np.sum(np.exp(func - func_max), axis=1, keepdims=True)
    y_true = np.zeros_like(probability)
    y_true[np.arange(num_train), y] = 1.0

    for i in range(num_train):
        for j in range(num_classes):
            loss -= y_true[i, j] * np.log(probability[i, j])
            dW_each[:, j] -= (y_true[i, j] - probability[i, j]) * X[i]
        dW += dW_each

    loss /= num_train
    loss += 1 / 2 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W: np.array, X: np.array, y: np.array, reg):
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train, _ = X.shape

    func = X.dot(W)
    func_max = np.reshape(np.max(func, axis=1), (num_train, 1))
    probability = np.exp(func - func_max) / np.sum(np.exp(func - func_max), axis=1, keepdims=True)

    y_true = np.zeros_like(probability)
    y_true[range(num_train), y] = 1.0

    loss += - np.sum(y_true * np.log(probability)) / num_train + 1 / 2 * reg * np.sum(W * W)
    dW += - np.dot(X.T, y_true - probability) / num_train + reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
