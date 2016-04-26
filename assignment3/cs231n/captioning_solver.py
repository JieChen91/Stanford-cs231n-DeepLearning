import numpy as np

from cs231n import optim
from cs231n.coco_utils import sample_coco_minibatch


class CaptioningSolver(object):
  """
  A CaptioningSolver encapsulates all the logic necessary for training
  image captioning models. The CaptioningSolver performs stochastic gradient
  descent using different update rules defined in optim.py.

  The solver accepts both training and validataion data and labels so it can
  periodically check classification accuracy on both training and validation
  data to watch out for overfitting.

  To train a model, you will first construct a CaptioningSolver instance,
  passing the model, dataset, and various options (learning rate, batch size,
  etc) to the constructor. You will then call the train() method to run the 
  optimization procedure and train the model.
  
  After the train() method returns, model.params will contain the parameters
  that performed best on the validation set over the course of training.
  In addition, the instance variable solver.loss_history will contain a list
  of all losses encountered during training and the instance variables
  solver.train_acc_history and solver.val_acc_history will be lists containing
  the accuracies of the model on the training and validation set at each epoch.
  
  Example usage might look something like this:
  
  data = load_coco_data()
  model = MyAwesomeModel(hidden_dim=100)
  solver = CaptioningSolver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  lr_decay=0.95,
                  num_epochs=10, batch_size=100,
                  print_every=100)
  solver.train()


  A CaptioningSolver works on a model object that must conform to the following
  API:

  - model.params must be a dictionary mapping string parameter names to numpy
    arrays containing parameter values.

  - model.loss(features, captions) must be a function that computes
    training-time loss and gradients, with the following inputs and outputs:

    Inputs:
    - features: Array giving a minibatch of features for images, of shape (N, D
    - captions: Array of captions for those images, of shape (N, T) where
      each element is in the range (0, V].

    Returns:
    - loss: Scalar giving the loss
    - grads: Dictionary with the same keys as self.params mapping parameter
      names to gradients of the loss with respect to those parameters.
  """

  def __init__(self, model, data, **kwargs):
    """
    Construct a new CaptioningSolver instance.
    
    Required arguments:
    - model: A model object conforming to the API described above
    - data: A dictionary of training and validation data from load_coco_data

    Optional arguments:
    - update_rule: A string giving the name of an update rule in optim.py.
      Default is 'sgd'.
    - optim_config: A dictionary containing hyperparameters that will be
      passed to the chosen update rule. Each update rule requires different
      hyperparameters (see optim.py) but all update rules require a
      'learning_rate' parameter so that should always be present.
    - lr_decay: A scalar for learning rate decay; after each epoch the learning
      rate is multiplied by this value.
    - batch_size: Size of minibatches used to compute loss and gradient during
      training.
    - num_epochs: The number of epochs to run for during training.
    - print_every: Integer; training losses will be printed every print_every
      iterations.
    - verbose: Boolean; if set to false then no output will be printed during
      training.
    """
    self.model = model
    self.data = data
    
    # Unpack keyword arguments
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.lr_decay = kwargs.pop('lr_decay', 1.0)
    self.batch_size = kwargs.pop('batch_size', 100)
    self.num_epochs = kwargs.pop('num_epochs', 10)

    self.print_every = kwargs.pop('print_every', 10)
    self.verbose = kwargs.pop('verbose', True)

    # Throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
      extra = ', '.join('"%s"' % k for k in kwargs.keys())
      raise ValueError('Unrecognized arguments %s' % extra)

    # Make sure the update rule exists, then replace the string
    # name with the actual function
    if not hasattr(optim, self.update_rule):
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(optim, self.update_rule)

    self._reset()


  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """
    # Set up some variables for book-keeping
    self.epoch = 0
    self.best_val_acc = 0
    self.best_params = {}
    self.loss_history = []
    self.train_acc_history = []
    self.val_acc_history = []

    # Make a deep copy of the optim_config for each parameter
    self.optim_configs = {}
    for p in self.model.params:
      d = {k: v for k, v in self.optim_config.iteritems()}
      self.optim_configs[p] = d


  def _step(self):
    """
    Make a single gradient update. This is called by train() and should not
    be called manually.
    """
    # Make a minibatch of training data
    minibatch = sample_coco_minibatch(self.data,
                  batch_size=self.batch_size,
                  split='train')
    captions, features, urls = minibatch

    # Compute loss and gradient
    loss, grads = self.model.loss(features, captions)
    self.loss_history.append(loss)

    # Perform a parameter update
    for p, w in self.model.params.iteritems():
      dw = grads[p]
      config = self.optim_configs[p]
      next_w, next_config = self.update_rule(w, dw, config)
      self.model.params[p] = next_w
      self.optim_configs[p] = next_config

  
  # Implement BLEU for caption evaluation.
  def check_accuracy(self, X, y, batch_size=100):
    """
    Check accuracy of the model on the provided data.
    
    Inputs:
    - X: Array of data, of shape (N, d_1, ..., d_k)
    - y: Array of captions, of shape (N, V)
    - batch_size: Split X and y into batches of this size to avoid using too
      much memory.
      
    Returns:
    - acc: Scalar giving the fraction of words that co-exist in prediction and 
      ground truth.
    """
    # return 0.0

    # The max length of each predicted caption, which is defined in model.sample()
    max_length = 30
    N = X.shape[0]
    y_pred = np.zeros((N, max_length))
    
    # Compute predictions in batches(Note: it's not stochaic batch here, it's in order.)
    num_batches = N / batch_size
    if N % batch_size != 0:
      num_batches += 1

    for i in xrange(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      y_pred[start:end, :] = self.model.sample(X[start:end, :], max_length=max_length)

    # BLEU
    # Don't count <NULL> token.
    len_y_pred = np.sum((y_pred != self.model._null), axis = 1)
    # Number of words in y_pred that "hit" groudtruth, y.
    hit_count = np.zeros_like(len_y_pred)
    for i in xrange(N):
      hit_y = [w for w in y_pred[i, :] if w in y[i, :] and w != self.model._null]
      # Remove duplicate elements to ensure acc is between [0, 1].
      hit_y = list(set(hit_y))
      for w in hit_y:
        hit_count[i] += np.minimum(y_pred[i, :].tolist().count(w), y[i, :].tolist().count(w))
    acc = np.mean(hit_count / len_y_pred)
    return acc


  def train(self):
    """
    Run optimization to train the model.
    """
    num_train = self.data['train_captions'].shape[0]
    iterations_per_epoch = max(num_train / self.batch_size, 1)
    num_iterations = self.num_epochs * iterations_per_epoch

    for t in xrange(num_iterations):
      self._step()

      # Maybe print training loss
      if self.verbose and t % self.print_every == 0:
        print '(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1])

      # At the end of every epoch, increment the epoch counter and decay the
      # learning rate.
      epoch_end = (t + 1) % iterations_per_epoch == 0
      if epoch_end:
        self.epoch += 1
        for k in self.optim_configs:
          self.optim_configs[k]['learning_rate'] *= self.lr_decay

      # Check train and val accuracy on the first iteration, the last
      # iteration, and at the end of each epoch.
      # TODO: Implement some logic to check Bleu on validation set periodically
      first_it = (t == 0)
      last_it = (t == num_iterations - 1)
      if first_it or last_it or epoch_end:
        # Subsample num_train samples for evaluation.
        train_captions, train_features, _ = sample_coco_minibatch(self.data,
                                              batch_size=num_train, split='train')
        val_captions, val_features, _  = sample_coco_minibatch(self.data,
                                           batch_size=num_train, split='val')

        train_acc = self.check_accuracy(train_features, train_captions)
        val_acc = self.check_accuracy(val_features, val_captions)
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)
   

        if self.verbose:
          print '(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                 self.epoch, self.num_epochs, train_acc, val_acc)

        # Keep track of the best model
        if val_acc > self.best_val_acc:
          self.best_val_acc = val_acc
          self.best_params = {}
          for k, v in self.model.params.iteritems():
            self.best_params[k] = v.copy()

    # At the end of training swap the best params into the model
    self.model.params = self.best_params

