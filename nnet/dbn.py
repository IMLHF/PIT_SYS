from .rbm import RBM
import os
import time

class DBN(object):
  def __init__(self, sizes,learning_rate=0.01,cd_k=1):
    self._sizes=sizes
    self._rbm_list=[]
    for i,size in enumerate(self._sizes[1:]):
      visible_size=self._sizes[i]
      hidden_size=size
      self._rbm_list.append(RBM(
        "RBM_%d" % i,visible_size,hidden_size,learning_rate=learning_rate,CDk=cd_k))

  def pretrain(self,vis,batch_size=128,n_epoches=10,verbose=True):
    for rbm in self._rbm_list:
      rbm.rbm_train(vis,batch_size,n_epoches,verbose)
      vis=rbm.rbm_forward(vis)
    return self.get_layer_reconstruct_err()

  def get_layer_reconstruct_err(self):
    err=[]
    for rbm in self._rbm_list:
      err.append(rbm.train_error[-1])
    return err
