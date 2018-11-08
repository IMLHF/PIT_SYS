import tensorflow as tf
import numpy as np
from .print_ import print_
import time
import os


class RBM(object):
  def __init__(self, name, isize, osize, learning_rate=0.01, CDk=1):
    self._name = name
    self._input_size = isize
    self._output_size = osize
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
      self._weight = tf.get_variable(self._name+"_weight", [self._input_size, self._output_size],
                                     initializer=tf.random_normal_initializer(dtype=tf.float64),
                                     dtype=tf.float64)
      self._v_bias = tf.get_variable(self._name+"_v_bias", [self._input_size],
                                     initializer=tf.random_normal_initializer(dtype=tf.float64),
                                     dtype=tf.float64)
      self._h_bias = tf.get_variable(self._name+"_h_bias", [self._output_size],
                                     initializer=tf.random_normal_initializer(dtype=tf.float64),
                                     dtype=tf.float64)
    self._learning_rate = learning_rate
    self._CDk = CDk
    self.session = None

    # 一定保证session.run()中的节点没有反复创建，否则很容易出现内存泄漏
    self.x_in = tf.placeholder(tf.float64, shape=[None, self._input_size])
    self.rbm_pretrain = self._rbm_train_epoche(self.x_in)
    self.x_loss = self.reconstruct(self.x_in)
    self.vis_for_up=tf.placeholder(tf.float64, shape=[None, self._input_size])
    self.x_up, _ = self._predict_h_given_v(self.vis_for_up)
    # os.system("nvidia-smi")
    # time.sleep(2)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.session = tf.Session(config=config)
    # time.sleep(5)
    # os.system("nvidia-smi")
    # time.sleep(5)
    self.session.run(tf.global_variables_initializer())
    # os.system("nvidia-smi")
    # time.sleep(5)

  def _0_1_sample_given_p(self, p):
    return tf.nn.relu(tf.sign(p - tf.random_uniform(tf.shape(p),dtype=tf.float64)))

  def _predict_h_given_v(self, v):
    prob = tf.nn.sigmoid(
        tf.matmul(v, self._weight)+self._h_bias)
    return prob, self._0_1_sample_given_p(prob)

  def _predict_v_given_h(self, h):
    prob = tf.nn.sigmoid(
        tf.matmul(h, tf.transpose(self._weight))+self._v_bias)
    return prob, self._0_1_sample_given_p(prob)

  def _CDk_f(self, vis):
    v0_prob = vis
    h0_prob, h0_sample = self._predict_h_given_v(v0_prob)
    hk_sample = h0_sample
    hk_prob = h0_sample
    for i in range(self._CDk):
      vk_prob, vk_sample = self._predict_v_given_h(hk_prob)  # 隐层使用概率
      # vk_prob, vk_sample = self._predict_v_given_h(hk_sample)  # 隐层使用逻辑单元
      hk_prob, hk_sample = self._predict_h_given_v(vk_prob)   # 可视层使用概率代替

    delta_w_positive = tf.matmul(tf.transpose(v0_prob), h0_prob)
    delta_w_negative = tf.matmul(tf.transpose(vk_prob), hk_prob)

    delta_w = tf.subtract(delta_w_positive, delta_w_negative) / \
        tf.to_double(tf.shape(v0_prob)[0])
    delta_vb = tf.reduce_mean(v0_prob-vk_prob, 0)
    delta_hb = tf.reduce_mean(h0_prob-hk_prob, 0)

    return delta_w, delta_vb, delta_hb

  def _rbm_train_epoche(self, vis):
    delta_w, delta_vb, delta_hb = self._CDk_f(vis)
    # update rbm parameters
    update_w_op = self._weight.assign_add(self._learning_rate*delta_w)
    update_vb_op = self._v_bias.assign_add(self._learning_rate*delta_vb)
    update_hb_op = self._h_bias.assign_add(self._learning_rate*delta_hb)

    return [update_w_op, update_vb_op, update_hb_op]

  def reconstruct(self, vis):
    _, h_samp = self._predict_h_given_v(vis)
    for i in range(self._CDk):
      v_recon, _ = self._predict_v_given_h(h_samp)
      _, h_samp = self._predict_h_given_v(v_recon)
    return tf.reduce_mean(tf.square(vis - v_recon))

  def rbm_train(self, data_x, batch_size=128, n_epoches=1, verbose=True):

    x_len = np.shape(data_x)[0]
    n_batches = x_len//batch_size if (x_len % batch_size == 0) else ((
          x_len//batch_size)+1)

    # # whether or not plot
    # if self.plot is True:
    #     plt.ion() # start the interactive mode of plot
    #     plt.figure(1)

    errs = []
    sess = self.session
    mean_cost = []
    for epoch in range(n_epoches):
      mean_cost = []
      for i_batche in range(n_batches):
        s_site = i_batche*batch_size
        if(s_site+batch_size <= len(data_x)):
          e_site = s_site+batch_size
        else:
          e_site = len(data_x)
        batch_x = data_x[s_site:e_site]
        cost = sess.run(self.x_loss, feed_dict={self.x_in: batch_x})  # loss在前
        sess.run(self.rbm_pretrain, feed_dict={self.x_in: batch_x})
        mean_cost.append(cost)
        # print(cost)
      errs.append(np.mean(mean_cost))
      if verbose:
        print_('%s Training : Epoch %04d lost %g' %
               (self._name, epoch, errs[-1]))
      # # plot ?
      # if plt.fignum_exists(1):
      #     plt.plot(range(epoch+1),errs,'-r')
    self.train_error = errs
    return errs

  def rbm_forward(self, vis):
    assert np.shape(vis)[1] == self._input_size
    return self.session.run(self.x_up,feed_dict={self.vis_for_up:vis})

  def get_param(self):
    return self.session.run(self._weight), self.session.run(self._v_bias), self.session.run(self._h_bias)
