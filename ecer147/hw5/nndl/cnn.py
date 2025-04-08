import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

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

    C,H,W = input_dim
    F = num_filters
    Hf = filter_size
    Wf = filter_size

    self.params["W1"] = np.random.randn(F,C,Hf,Wf) * weight_scale
    self.params["b1"] = np.zeros(F)

    conv_out_H = 1 + (H + 2 * ((filter_size - 1) // 2) - Hf) // 1
    conv_out_W = 1 + (W + 2 * ((filter_size -1)// 2) - Wf) // 1

    pool_out_H = conv_out_H // 2
    pool_out_W = conv_out_W // 2

    self.params["W2"] = np.random.randn(F * pool_out_H * pool_out_W, hidden_dim) * weight_scale
    self.params["b2"] = np.zeros(hidden_dim)
    self.params["W3"] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params["b3"] = np.zeros(num_classes)

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
    conv_out, conv_cache = conv_relu_pool_forward(X,W1,b1, conv_param, pool_param)

    affine_relu_out, affine_relu_cache = affine_relu_forward(conv_out, W2, b2)

    scores, affine_cache = affine_forward(affine_relu_out, W3, b3)

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

    # loss and derivative of scores.
    loss,dscores=softmax_loss(scores,y)

    loss+=0.5*self.reg*(np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2))
    
    daffine_relu_out, grads["W3"],grads["b3"]=affine_backward(dscores,affine_cache)

    dconv_out, grads["W2"], grads["b2"]=affine_relu_backward(daffine_relu_out, affine_relu_cache)

    dx, grads["W1"], grads["b1"]=conv_relu_pool_backward(dconv_out, conv_cache)

    grads["W1"]+=self.reg*W1
    grads["W2"]+=self.reg*W2
    grads["W3"]+=self.reg*W3
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  


class NnForLast(object):
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=128, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=True):

        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        C, H, W = input_dim
        F = num_filters
        Hf = filter_size
        Wf = filter_size

        self.params["W1"] = np.random.randn(F, C, Hf, Wf) * weight_scale
        self.params["b1"] = np.zeros(F)
        
        if self.use_batchnorm:
            self.params["gamma1"] = np.ones(F)
            self.params["beta1"] = np.zeros(F)

        self.params["W2"] = np.random.randn(F * 2, F, Hf, Wf) * weight_scale
        self.params["b2"] = np.zeros(F * 2)
        
        if self.use_batchnorm:
            self.params["gamma2"] = np.ones(F * 2)
            self.params["beta2"] = np.zeros(F * 2)

        self.params["W3"] = np.random.randn(F * 4, F * 2, Hf, Wf) * weight_scale
        self.params["b3"] = np.zeros(F * 4)
        
        if self.use_batchnorm:
            self.params["gamma3"] = np.ones(F * 4)
            self.params["beta3"] = np.zeros(F * 4)

        pool_out_H = H // 8  
        pool_out_W = W // 8
        
        self.params["W4"] = np.random.randn(F * 4 * pool_out_H * pool_out_W, hidden_dim) * weight_scale
        self.params["b4"] = np.zeros(hidden_dim)
        
        self.params["W5"] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params["b5"] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    def forward(self, X):
        print(X.shape)

        out = X
        cache = {}
        out, cache['conv1'] = conv_relu_pool_forward(out, self.params['W1'], self.params['b1'], conv_param={'stride': 1, 'pad': 1}, pool_param={'pool_height': 2, 'pool_width': 2, 'stride': 2})

        if self.use_batchnorm:
            n,c,h,w = out.shape
            out =out.transpose(0,2,3,1).reshape(n*h*w,c)
            out, cache['bn1'] = batchnorm_forward(out, self.params['gamma1'], self.params['beta1'], bn_param={'mode': 'train'})
            out = out.reshape(n,h,w,c).transpose(0,3,1,2)

        out, cache['conv2'] = conv_relu_pool_forward(out, self.params['W2'], self.params['b2'], conv_param={'stride': 1, 'pad': 1}, pool_param={'pool_height': 2, 'pool_width': 2, 'stride': 2})
        if self.use_batchnorm:
            n,c,h,w = out.shape
            out =out.transpose(0,2,3,1).reshape(n*h*w,c)
            out, cache['bn2'] = batchnorm_forward(out, self.params['gamma2'], self.params['beta2'], bn_param={'mode': 'train'})
            out = out.reshape(n,h,w,c).transpose(0,3,1,2)

        out, cache['conv3'] = conv_relu_pool_forward(out, self.params['W3'], self.params['b3'], conv_param={'stride': 1, 'pad': 1}, pool_param={'pool_height': 2, 'pool_width': 2, 'stride': 2})


        if self.use_batchnorm:
            n,c,h,w = out.shape
            out =out.transpose(0,2,3,1).reshape(n*h*w,c)
            out, cache['bn3'] = batchnorm_forward(out, self.params['gamma3'], self.params['beta3'], bn_param={'mode': 'train'})
            out = out.reshape(n,h,w,c).transpose(0,3,1,2)

        out = out.reshape(out.shape[0], -1)
        

        out, cache['fc1'] = affine_forward(out, self.params['W4'], self.params['b4'])
        out = relu_forward(out)


        out, cache['fc2'] = affine_forward(out, self.params['W5'], self.params['b5'])
        
        return out, cache
    
    def backward(self, dout, cache):

        grads = {}
        
        dout, grads['W5'], grads['b5'] = affine_backward(dout, cache['fc2'])

        dout = relu_backward(dout, cache['fc1'])
        dout, grads['W4'], grads['b4'] = affine_backward(dout, cache['fc1'])

        dout, grads['W3'], grads['b3'] = conv_relu_pool_backward(dout, cache['conv3'])
        if self.use_batchnorm:
            dout, grads['gamma3'], grads['beta3'] = batchnorm_backward(dout, cache['bn3'])
        
        dout, grads['W2'], grads['b2'] = conv_relu_pool_backward(dout, cache['conv2'])
        if self.use_batchnorm:
            dout, grads['gamma2'], grads['beta2'] = batchnorm_backward(dout, cache['bn2'])

        dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, cache['conv1'])
        if self.use_batchnorm:
            dout, grads['gamma1'], grads['beta1'] = batchnorm_backward(dout, cache['bn1'])
        
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        grads['W3'] += self.reg * self.params['W3']
        grads['W4'] += self.reg * self.params['W4']
        grads['W5'] += self.reg * self.params['W5']
        
        return grads
    
    def loss(self, X, y=None):

      scores, cache = self.forward(X)


      if y is None:
        return scores
    
      loss, dscores = softmax_loss(scores, y)
      grads = self.backward(dscores, cache)
      loss += 0.5 * self.reg * sum(np.sum(w**2) for w in self.params.values() if w.ndim == 4)
      grads['W1'] += self.reg * self.params['W1']
      grads['W2'] += self.reg * self.params['W2']
      grads['W3'] += self.reg * self.params['W3']
      grads['W4'] += self.reg * self.params['W4']
      grads['W5'] += self.reg * self.params['W5']

      return loss, grads


  
  

