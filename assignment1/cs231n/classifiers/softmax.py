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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  for i in xrange(num_train):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores)
    sum_exp_scores = sum(exp_scores)
    loss += -np.log(exp_scores[y[i]] / sum_exp_scores)
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += (exp_scores[j] / sum_exp_scores - 1) * X[i]
      else:
        dW[:, j] += exp_scores[j] / sum_exp_scores * X[i]
    
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  N, D = X.shape
  num_classes = W.shape[1]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO(Not Finished, only achieving half-vectorized in dW computaion):      #
  # Compute the softmax loss and its gradient using no explicit loops.        #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  sumExpScores_eachX = np.sum(exp_scores, axis = 1)
  target_expScores = exp_scores[range(N), y]
  loss = np.sum(-np.log(target_expScores / sumExpScores_eachX)) / N + 0.5 * reg *np.sum(W * W)

  for i in xrange(N):
    scoreFactor = exp_scores[i, :] / sumExpScores_eachX[i]
    dW += X[i].reshape(D, 1).dot(scoreFactor.reshape(1, num_classes))
    dW[:, y[i]] -= X[i]
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

