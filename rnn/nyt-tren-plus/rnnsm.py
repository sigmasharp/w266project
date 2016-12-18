#***** Based on Assignment 1, Part 2 code, tailored for sentiment ******
import time

import tensorflow as tf
import numpy as np


def matmul3d(X, W):
  """Wrapper for tf.matmul to handle a 3D input tensor X.  Will perform multiplication along the last dimension.
     Args:X: [m,n,k]W: [k,l]  Returns:XW: [m,n,l]
  """
  Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
  XWr = tf.matmul(Xr, W)
  newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
  return tf.reshape(XWr, newshape)

def MakeFancyRNNCell(H, keep_prob, num_layers=1):
  """Make a fancy RNN cell.  Use tf.nn.rnn_cell functions to construct an LSTM cell.  Initialize forget_bias=0.0 for better training.
  Args:    H: hidden state size    keep_prob: dropout keep prob (same for input and output)   num_layers: number of cell layers
  Returns:    (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
  """
  cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0)
  cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob,
                                       output_keep_prob=keep_prob)
  cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
  return cell

class RNNSM(object):

  def __init__(self, V, Z, H, num_layers=1):
    """Init function.
    This function just stores hyperparameters. Real graphconstruction in the Build*Graph() functions below.
    Args: V: vocabulary size, Z: classifier size, H: hidden state dimension, num_layers: number of RNN layers
    """
    self.V = V
    self.H = H
    self.Z = Z
    self.num_layers = num_layers
    # Training hyperparameters; to be changed with feed_dict during training.
    with tf.name_scope("Training_Parameters"):
      self.learning_rate_ = tf.constant(0.1, name="learning_rate")

      self.dropout_keep_prob_ = tf.placeholder(tf.float32, name="dropout_keep_prob")
      
      # For gradient clipping, in case
      # Due to a bug in TensorFlow, this needs to be an ordinary python
      # constant instead of a tf.constant.
      self.max_grad_norm_ = 5.0

  def BuildCoreGraph(self):
    """Construct the core RNNSM graph for any use of the model.
    """
    # Input ids, with dynamic shape depending on input.
    # Should be shape [batch_size, max_time] and contain integer word indices.
    self.input_w_ = tf.placeholder(tf.int32, [None, None], name="w")
    
    # target_y_ the sentiment score+
    self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")
    
    # Get dynamic shape info from inputs
    with tf.name_scope("batch_size"):
      self.batch_size_ = tf.shape(self.input_w_)[0]
    with tf.name_scope("max_time"):
      self.max_time_ = tf.shape(self.input_w_)[1]

    # Get sequence length from input_w_, a vector with elements ns[i] = len(input_w_[i])
    # Can be overriden  in feed_dict for a different-length sequences in the same batch
    self.ns_ = tf.tile([self.max_time_], [self.batch_size_,], name="ns")
    
    # seed
    self.seed = 0

    # Construct embedding layer, from V x H to H
    
    # Wem of shape(V, H), the embedding layer weight matrix, or the lookup table
    self.Wem_ = tf.Variable(tf.random_uniform([self.V, self.H], minval=-1.0, maxval=1.0, seed=0), name="Wem")
    
    # the input_x_ of shape(b, t, H), the input vector, from the lookup of intput_w_ of the shape [b, t], 
    # an index int between 0 and V - 1 against the lookup table Wem,
    # i don't think the inner reshape to the input_w_ is necessary, but i did it anyway
    self.input_x_ = tf.reshape(tf.nn.embedding_lookup(params=(self.Wem_), 
        ids=self.input_w_), [self.batch_size_, self.max_time_, self.H], name="x")
    
    # Construct RNN/LSTM cell and recurrent layer   
    # the fancy cell, a LSTM cell (with 4 affine layers)
    self.cell_ = MakeFancyRNNCell(self.H, self.dropout_keep_prob_, self.num_layers)
    
    # initial_h_ of shape(b, t, H), the initial hidden state for the rnn layer
    # initialize it to all zeros at the beginning
    self.initial_h_ = self.cell_.zero_state(self.batch_size_, dtype=tf.float32)

    # ouput of shape(b, t, H), the output state or vector from the rnn layer, or the input to the output layer
    # final_h of shape(b, t, H), the final state from the rnn layer
    self.output_, self.final_h_ = tf.nn.dynamic_rnn(self.cell_, inputs=self.input_x_, 
        sequence_length=self.ns_, initial_state=self.initial_h_, dtype=tf.float32)
   
    # Softmax output layer, over sentiment+
    
    # Wout of shape(H, Z), the output layer weight matrix, and 
    # bout of shape(Z), the output layer bias vector
    self.Wout_ = tf.Variable(tf.random_uniform([self.H, self.Z], minval=-1.0, maxval=1.0, seed=self.seed), name="Wout")
    self.bout_ = tf.Variable(tf.zeros([self.Z]), dtype=tf.float32, name="b")
    
    # logits of shape(b, t, Z), the logits from the output layer, for the whole RNNLM
    self.logits_ = tf.reshape(matmul3d(self.output_, self.Wout_) + self.bout_, [self.batch_size_, self.max_time_, self.Z])
    
    # y^hat of shape(b, t), the softmax from the logits
    self.y_hat_ = tf.reshape(tf.argmax(tf.reshape(self.logits_, [-1, self.Z]), 1, name="y_hat"), [self.batch_size_, self.max_time_])
    
    # Loss computation (true loss, for prediction)
    # loss of shape (), a scalar of the true loss, or the sum of the cross entrypy loss over the logits
    self.loss_ = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_, self.target_y_, name = "loss"))    

  def BuildTrainGraph(self):
    """Construct the training ops.
    - train_loss_ (optional): an approximate loss function for training if sentiment includes vacabulary
    - train_step_ : a training op that can be called once per batch
    Loss function returns a *scalar* value representing the _summed_ loss across all examples in the batch 
    """
    # Define loss function(s)
    with tf.name_scope("Train_Loss"):
      # **** when the target is only sentiment, 6-bit id, train_loss should be the same as true loss
     
      #  train_loss of shape(), the sum of a sampled softmax loss over the sampled, only when sentiment is combined with V
      #self.num_sampled = 100
      #self.train_loss_ = tf.reduce_sum(tf.nn.sampled_softmax_loss(tf.transpose(self.Wout_), self.bout_, 
      #    tf.reshape(self.output_, [-1, self.H]), labels=tf.reshape(self.target_y_, [-1,1]), num_sampled=self.num_sampled, 
      #    num_classes=self.Z, name='train_loss'))
      self.train_loss_ = self.loss_
      
    # Define optimizer and training op
    with tf.name_scope("Training"):
        self.optimizer_ = tf.train.AdagradOptimizer(self.learning_rate_)
        self.train_step_ = self.optimizer_.minimize(self.train_loss_)
    
  def BuildSamplerGraph(self):
    """Construct the sampling ops.
    pred_samples_ is a Tensor of integer indices for sampled predictions for each batch element, at each timestep.
    """
    #self.pred_proba_ = tf.nn.softmax(self.logits_, name="pred_proba")
    #self.pred_max_ = tf.argmax( self. logits_, 1, name="pred_max")

    self.pred_samples_ = tf.reshape(tf.multinomial(tf.reshape(self.logits_, [-1, self.Z]), 1), [self.batch_size_, self.max_time_, 1])