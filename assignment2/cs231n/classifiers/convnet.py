import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  Arbitrary architecture convnet
  """

  def __init__(self, input_dim=(3, 32, 32), layers=[], weight_scale=1e-3, reg=0.0, dropout=0.0, num_classes=10, dtype=np.float32):
    """
    Initialize the network by given params

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - layers: List of tuples [('layername1', params1), ..., ('layernameN', paramsN)]
    - weight_scale: Initiate the scale of W in 'conv'
    - reg: Scalar giving L2 regularization strength
    - dropout: The probability of keeping a nueron.
    - num_classes: Number of scores to produce from the final affine layer.
    - dtype: numpy datatype to use for computation.
    """

    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.layers = layers

    # The dropout_param and pool_param are the same for every dropout/maxpool layer.
    self.dropout_param = {'mode': 'train', 'p': dropout}
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # The input dimension of each layer is dynamically changing.
    dimension = input_dim
    # Counters for conv/affine and bachnorm.
    cv_a_ind = 0
    bn_ind = 0
    for l, layer in enumerate(self.layers):
      layertype = layer[0]
      layerparams = layer[1] if len(layer) > 1 else None

      # initialize params for every layer by its type
      # skip relu -- nothing to init
      if layertype == 'conv':
        cv_a_ind += 1
        depth, size = layerparams
        shape = (depth, dimension[-3], size, size)
        self.params['W' + str(cv_a_ind)] = weight_scale * np.random.randn(*shape).astype(dtype)
        self.params['b' + str(cv_a_ind)] = np.zeros(depth).astype(dtype)
        dimension = (depth, dimension[-2], dimension[-1])

      if layertype == 'affine':
        cv_a_ind += 1
        dim = layerparams
        shape = (np.prod(dimension), dim)
        self.params['W' + str(cv_a_ind)] = weight_scale * np.random.randn(*shape).astype(dtype)
        self.params['b' + str(cv_a_ind)] = np.zeros(dim).astype(dtype)
        dimension = dim

      if layertype == 'spatial_batchnorm':
        bn_ind += 1
        self.params['gamma' + str(bn_ind)] = np.ones(dimension[-3])
        self.params['beta' + str(bn_ind)] = np.zeros(dimension[-3])

      if layertype == 'batchnorm':
        bn_ind += 1
        self.params['gamma' + str(bn_ind)] = np.ones(dimension)
        self.params['beta' + str(bn_ind)] = np.zeros(dimension)

      if layertype == 'pool':
        dimension = (dimension[0], dimension[1] / 2, dimension[2] / 2)

    # Last affine before softmax
    cv_a_ind += 1
    shape = (np.prod(dimension), num_classes)
    self.params['W' + str(cv_a_ind)] = weight_scale * np.random.randn(*shape).astype(dtype)
    self.params['b' + str(cv_a_ind)] = np.zeros(num_classes).astype(dtype)

    # Different from dropout_param, bn_params are different among different batchnorm layers.
    self.bn_params = [{'mode': 'train', 'momentum': 0.9} for i in xrange(bn_ind)]

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient
    """

    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'
    self.dropout_param['mode'] = mode
    for bn_param in self.bn_params:
      bn_param[mode] = mode

    # FORWARD PASS
    h, caches = X, []
    cv_a_ind = 0
    bn_ind = 0
    for l, layer in enumerate(self.layers):
      layertype = layer[0]
      layerparams = layer[1] if len(layer) > 1 else None

      if layertype == 'conv':
        cv_a_ind += 1
        W, b = self.params['W' + str(cv_a_ind)], self.params['b' + str(cv_a_ind)]
        # In general, the conv does not change the size by padding.
        filter_size = W.shape[-1]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        h, c = conv_forward_fast(h, W, b, conv_param)
        caches.append(c)

      if layertype == 'affine':
        cv_a_ind += 1
        W, b = self.params['W' + str(cv_a_ind)], self.params['b' + str(cv_a_ind)]
        h, c = affine_forward(h, W, b)
        caches.append(c)

      if layertype == 'relu':
        h, c = relu_forward(h)
        caches.append(c)

      if layertype == 'pool': 
        h, c = max_pool_forward_fast(h, self.pool_param)
        caches.append(c)

      if layertype == 'spatial_batchnorm':
        bn_ind += 1
        gamma, beta = self.params['gamma' + str(bn_ind)], self.params['beta' + str(bn_ind)]
        h, c = spatial_batchnorm_forward(h, gamma, beta, self.bn_params[bn_ind - 1])      
	caches.append(c)

      if layertype == 'batchnorm':
        bn_ind += 1
        gamma, beta = self.params['gamma' + str(bn_ind)], self.params['beta' + str(bn_ind)]
        h, c = batchnorm_forward(h, gamma, beta, self.bn_params[bn_ind - 1])
        caches.append(c)

      if layertype == 'dropout':
        h, c = dropout_forward(h, self.dropout_param)
        caches.append(c)


    # extra affine
    cv_a_ind += 1
    W, b = self.params['W' + str(cv_a_ind)], self.params['b' + str(cv_a_ind)]
    h, c = affine_forward(h, W, b)
    caches.append(c)

    # LOSS COMPUTATION
    scores = h
    if y is None:
      return scores

    loss, grads = 0, {}
    loss, ds = softmax_loss(scores, y)
    for ind in xrange(cv_a_ind):
      loss += 0.5 * self.reg * np.sum(self.params['W' + str(ind + 1)] ** 2)

    # BACKWARD PASS
    # extra affine
    c = caches.pop()
    da, dW, db = affine_backward(ds, c)
    grads['b' + str(cv_a_ind)] = db
    W = self.params['W' + str(cv_a_ind)]
    dW += self.reg * W
    grads['W' + str(cv_a_ind)] = dW
    cv_a_ind -= 1

    dtemp = da
    for l, layer in enumerate(reversed(self.layers)):
      layertype = layer[0]

      if layertype == 'conv':
        c = caches.pop()
        dtemp, dW, db = conv_backward_fast(dtemp, c)
        grads['b' + str(cv_a_ind)] = db
        W = self.params['W' + str(cv_a_ind)]
        dW += self.reg * W
        grads['W' + str(cv_a_ind)] = dW
        cv_a_ind -= 1

      if layertype == 'affine':
        c = caches.pop()    
	dtemp, dW, db = affine_backward(dtemp, c)
        grads['b' + str(cv_a_ind)] = db
	W = self.params['W' + str(cv_a_ind)]
	dW += self.reg * W
	grads['W' + str(cv_a_ind)] = dW
        cv_a_ind -= 1

      if layertype == 'relu':
        c = caches.pop()
        dtemp = relu_backward(dtemp, c)

      if layertype == 'pool':
        c = caches.pop()
        dtemp = max_pool_backward_fast(dtemp, c)

      if layertype == 'spatial_batchnorm':
	c = caches.pop()
	dtemp, dgamma, dbeta = spatial_batchnorm_backward(dtemp, c)
        grads['gamma' + str(bn_ind)] = dgamma
        grads['beta' + str(bn_ind)] = dbeta
        bn_ind -= 1

      if layertype == 'batchnorm':
        c = caches.pop()
        dtemp, dgamma, dbeta = batchnorm_backward(dtemp, c)
        grads['gamma' + str(bn_ind)] = dgamma
        grads['beta' + str(bn_ind)] = dbeta
        bn_ind -= 1
    
      if layertype == 'dropout':
        c = caches.pop()
        dtemp = dropout_backward(dtemp, c)

    return loss, grads
