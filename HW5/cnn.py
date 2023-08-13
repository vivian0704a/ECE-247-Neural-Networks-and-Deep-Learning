# cnn.py

import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(0, weight_scale, [num_filters, C, filter_size, filter_size])
    self.params['b1'] = np.zeros([num_filters])
    stride = 1
    pad = (filter_size - 1) / 2    
    h_out = (H - filter_size + 2*pad)/stride + 1
    w_out = (W - filter_size + 2*pad)/stride + 1
    h_pool = int((h_out - 2)/2) + 1
    w_pool = int((w_out - 2)/2) + 1
    self.params['W2'] = np.random.normal(0, weight_scale, [h_pool*w_pool*num_filters, hidden_dim])
    self.params['b2'] = np.zeros([hidden_dim])
    self.params['W3'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
    self.params['b3'] = np.zeros([num_classes])

    if self.use_batchnorm:
      self.params['gamma1'] = np.ones(num_filters)
      self.params['beta1'] = np.zeros(num_filters)
      self.params['gamma2'] = np.ones(hidden_dim)
      self.params['beta2'] = np.zeros(hidden_dim)
      self.bn_params = [{'mode': 'train'}, {'mode': 'train'}]

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    
    if self.use_batchnorm:
      if y is None:
        mode = 'test'
      else: mode = 'train'
      for n in self.bn_params:
        n['mode'] = mode
      gamma1, gamma2, beta1, beta2 = self.params['gamma1'], self.params['gamma2'], self.params['beta1'], self.params['beta2']
      pool_out, pool_cache = conv_bn_relu_pool_forward(X, W1, b1, conv_param, gamma1, beta1, self.bn_params[0], pool_param)
      relu_out, relu_cache = affine_batchnorm_relu_forward(pool_out, W2, b2, gamma2, beta2, self.bn_params[1])
    else:
      pool_out, pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
      relu_out, relu_cache = affine_relu_forward(pool_out, W2, b2)
    scores, cache = affine_forward(relu_out, W3, b3)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    loss, dscores = softmax_loss(scores, y)
    loss += 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
    dx3, grads['W3'], grads['b3'] = affine_backward(dscores, cache)
    if self.use_batchnorm:
      dx2, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = affine_batchnorm_relu_backward(dx3, relu_cache)
      dx1, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_bn_relu_pool_backward(dx2, pool_cache)
    else: 
      dx2, grads['W2'], grads['b2'] = affine_relu_backward(dx3, relu_cache)
      dx1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx2, pool_cache)
    grads['W3'] = grads['W3'] + self.reg*W3 
    grads['W2'] = grads['W2'] + self.reg*W2 
    grads['W1'] = grads['W1'] + self.reg*W1 

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
