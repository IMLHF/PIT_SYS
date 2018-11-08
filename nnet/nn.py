import tensorflow as tf
import numpy as np
from .print_ import print_
from PIL import Image


class MLP1(object):
  def __init__(
          self, sizes=None, learning_rate=None, batch_size=None, n_epoches=None):
    self._sizes = sizes
    self._learning_rate = learning_rate
    self._batch_size = batch_size
    self._n_epoches = n_epoches
    self.w_list = []
    self.b_list = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.session = tf.Session(config=config)
    if self._sizes is None:
      return
    for i, size in enumerate(self._sizes[1:]):
      in_size = self._sizes[i]
      out_size = size
      self.w_list.append(tf.get_variable("nn_weight_"+str(i), [in_size, out_size],
                                         dtype=tf.float64,
                                         initializer=tf.random_normal_initializer()))
      # initializer=tf.constant_initializer(self.w_list[i])))
      self.b_list.append(tf.get_variable("nn_bias_"+str(i), [out_size],
                                         dtype=tf.float64,
                                         initializer=tf.random_normal_initializer()))
      # initializer=tf.constant_initializer(self.b_list[i])))

    self.x_input = tf.placeholder("float64", [None, self._sizes[0]])
    self.y_target = tf.placeholder("float64", [None, self._sizes[-1]])
    self.y_output = tf.nn.sigmoid(
        self._MLP(self.x_input, self.w_list, self.b_list))
    # TODO fix loss fun
    # loss_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     logits=y_output, labels=y_target))
    self.loss_cross_entropy = tf.reduce_mean(
        tf.square(tf.subtract(tf.multiply(self.y_output, self.x_input), self.y_target)))
    self.train_op = tf.train.GradientDescentOptimizer(
        learning_rate=self._learning_rate).minimize(self.loss_cross_entropy)

    self.mean_error_square_y = tf.reduce_mean(
        tf.cast(tf.square(tf.subtract(self.y_output, self.y_target)), tf.float32))
    self.mean_error_square_mask = tf.reduce_mean(
        tf.square(tf.subtract(tf.multiply(self.y_output, self.x_input), self.y_target)))
    self.logical_accuracy_rate = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(self.y_output, 1), tf.argmax(self.y_target, 1)), tf.float32))
    self.session.run(tf.global_variables_initializer())

  def load_from_dbn_to_normalNN(self, dbn):
    assert len(self._sizes) == len(dbn._sizes)
    for i in range(len(dbn._sizes)):
      assert dbn._sizes[i] == self._sizes[i]
    for i in range(len(dbn._sizes)-1):
      self.w_list[i], _, self.b_list[i] = dbn._rbm_list[i].get_param()

  def load_from_dbn_to_reconstructNN(self, dbn):
    assert len(self._sizes) == len(dbn._sizes)*2-1
    for i in range(len(dbn._sizes)):
      assert dbn._sizes[i] == self._sizes[i]
      assert self._sizes[-i-1] == dbn._sizes[i]
    for i in range(len(dbn._sizes)-1):
      self.w_list[i], _, self.b_list[i] = dbn._rbm_list[i].get_param()
    n_dbn_layers = len(dbn._sizes)
    for i in range(n_dbn_layers-1):
      w, vb, _ = dbn._rbm_list[i].get_param()
      self.w_list[-i-1] = np.transpose(w)
      self.b_list[-i-1] = vb

  # start_layer，end_layer : 网络层编号，从0开始编号。
  # 从nnet中抽取start_layer到end_layer层作为新的网络
  def load_layers_from_NN(self, nnet, start_layer, end_layer):
    assert start_layer < end_layer
    assert end_layer-start_layer+1 < len(nnet._sizes)
    self._sizes = nnet._sizes[start_layer:end_layer+1]
    self._learning_rate = nnet._learning_rate
    self._batch_size = nnet._batch_size
    self._n_epoches = nnet._n_epoches
    self.w_list = nnet.w_list[start_layer:end_layer]
    self.b_list = nnet.b_list[start_layer:end_layer]

  def _load_w_b_from_self(self):
    pass
    # w_list = []
    # b_list = []
    # for i, size in enumerate(self._sizes[1:]):
    #   in_size = self._sizes[i]
    #   out_size = size
    #   w_list.append(tf.get_variable("nn_weight_"+str(i), [in_size, out_size],
    #                                 dtype=tf.float64,
    #                                 # initializer=tf.random_normal_initializer()))
    #                                 initializer=tf.constant_initializer(self.w_list[i])))
    #   b_list.append(tf.get_variable("nn_bias_"+str(i), [out_size],
    #                                 dtype=tf.float64,
    #                                 # initializer=tf.random_normal_initializer()))
    #                                 initializer=tf.constant_initializer(self.b_list[i])))
    # return w_list, b_list

  def _save_w_b_to_self(self, session, w_list, b_list):
    self.w_list = session.run(w_list)
    self.b_list = session.run(b_list)

  def _MLP(self, x_in, w_list, b_list):
    y_out = x_in
    for i in range(len(self._sizes)-2):
      y_out = tf.nn.sigmoid(
          tf.add(tf.matmul(y_out, w_list[i]), b_list[i]))
    y_out = tf.add(
        tf.matmul(y_out, w_list[len(self._sizes)-2]), b_list[len(self._sizes)-2])  # TODO fix loss fun
    return y_out

  def train(self, X, Y, verbose=True):
    batch_size = self._batch_size
    n_epoches = self._n_epoches
    display_epoches = 1

    session_t = self.session
    for epoch in range(n_epoches):
      avg_lost = 0.0
      x_len=len(X)
      total_batch = x_len//batch_size if (x_len % batch_size == 0) else ((
          x_len//batch_size)+1)
      for i in range(total_batch):
        s_site = i*batch_size
        if(s_site+batch_size <= len(X)):
          e_site = s_site+batch_size
        else:
          e_site = len(X)
        x_batch = X[s_site:e_site]
        y_batch = Y[s_site:e_site]
        _, lost_t = session_t.run([self.train_op, self.loss_cross_entropy],
                                  feed_dict={
            self.x_input: x_batch,
            self.y_target: y_batch
        })
        self._save_w_b_to_self(session_t, self.w_list, self.b_list)

        avg_lost += float(lost_t)/total_batch
      if (epoch % display_epoches == 0) and verbose:
        print_("NNET Training : Epoch"+' %04d' %
               (epoch+1)+" Lost "+str(avg_lost))
    if verbose:
      print_("Optimizer Finished!")
    return avg_lost

  def test(self, X, Y):
    return str(self.session.run(self.mean_error_square_mask,
                                feed_dict={self.x_input: X,
                                           self.y_target: Y}))

  def test_linear(self, X, Y):
    print_("Error: "+str(self.session.run(self.mean_error_square_y,
                                          feed_dict={self.x_input: X,
                                                     self.y_target: Y})))

  def test_logical(self, X, Y):
    print_("Accuracy: " + str(self.session.run(self.logical_accuracy_rate,
                                               feed_dict={self.x_input: X,
                                                          self.y_target: Y})))

  def predict(self, X):
    __predict = self.session.run([self.y_output],
                                 feed_dict={self.x_input: X})
    return __predict
