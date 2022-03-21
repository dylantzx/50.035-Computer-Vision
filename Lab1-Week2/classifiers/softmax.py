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
    # https://ljvmiranda921.github.io/notebook/2017/02/14/softmax-classifier/
    num_of_examples = X.shape[0]
    num_of_classes = W.shape[1]
    
    for i in range(num_of_examples):

      f_i = X[i].dot(W)
      f_i -= np.max(f_i)

      sum_of_exp = np.sum(np.exp(f_i))
      p = lambda k: np.exp(f_i[k]) / sum_of_exp
      loss += -np.log(p(y[i]))

      for k in range(num_of_classes):
        prob_k = p(k)
        dW[:, k] += (prob_k - (k == y[i])) * X[i]

    loss = loss/num_of_examples + 0.5 *reg * np.sum(W * W)
    dW = dW/num_of_classes + reg*W

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
    # https://ljvmiranda921.github.io/notebook/2017/02/14/softmax-classifier/
    num_of_examples = X.shape[0]
    num_of_classes = W.shape[1]

    f = X.dot(W)
    f -= np.max(f, axis=1, keepdims=True)
    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    p = np.exp(f)/sum_f

    loss = np.sum(-np.log(p[np.arange(num_of_examples), y]))

    ind = np.zeros_like(p)
    ind[np.arange(num_of_examples), y] = 1
    dW = X.T.dot(p - ind)

    loss = loss/num_of_examples + 0.5 *reg * np.sum(W * W)
    dW = dW/num_of_classes + reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW

