import data_prepare
from nnet.mlp1 import MLP1
from nnet.dbn import DBN
import scipy.io
import utils
import numpy as np
from nnet.print_ import print_,print_f
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

if __name__ == "__main__":
  data_prepare.run()

  _dbn = DBN([257*7, 2048, 2048, 2048, 257*2])
  n_epoches_pertrainf = 5
  err_along_epoch = []
  for i in range(n_epoches_pertrainf):
    mixed_wav_list = scipy.io.loadmat(
        "data/train/mixed_wav_dir.mat")["mixed_wav_dir"]
    err_along_utt = []
    for mixed_wavs in mixed_wav_list:
      utt1 = mixed_wavs[0]
      utt2 = mixed_wavs[1]
      x = utils.prepare_x_for_rbm(utt1, utt2, 7)
      # print(x)
      # exit(0)
      # region
      # print(np.max(x),np.min(x))
      # x_,y_=utils.prepare_x_y_for_dnn(utt1,utt2,7)
      # print(np.max(x_),np.min(x_),np.max(y_),np.min(y_))
      # exit(0)
      # print(np.shape(x),np.shape(x_),np.shape(y_))
      # exit(0)
      # print(np.shape(x))
      # print(x.dtype)
      # print(type(x[0][0]))
      # endregion
      # os.system("nvidia-smi")
      # time.sleep(5)
      layer_err = _dbn.pretrain(x, batch_size=2048, n_epoches=10, verbose=True)
      err_along_utt.append(layer_err)
      print_f(mixed_wavs,"log/RBM_layer_err.log")
      print_f("RBM layer err : "+str(layer_err),"log/RBM_layer_err.log")
    err_along_epoch.append(err_along_utt)

  _nn = MLP1([257*7, 2048, 2048, 2048, 257*2],
             learning_rate=0.01, batch_size=2048, n_epoches=1)
  _nn.load_from_dbn_to_normalNN(_dbn)
  n_epoches_train = 20
  err_list_train = []
  err_list_val = []
  for i in range(n_epoches_train):
    mixed_wav_list = scipy.io.loadmat(
        "data/train/mixed_wav_dir.mat")["mixed_wav_dir"]
    err_epoch = 0
    for mixed_wavs in mixed_wav_list:
      utt1 = mixed_wavs[0]
      utt2 = mixed_wavs[1]
      x_, y_ = utils.prepare_x_y_for_dnn(utt1, utt2, 7)
      err = _nn.train(x_, y_, verbose=False)
      err_epoch += (err/len(mixed_wav_list))
    err_epoch /= n_epoches_train
    err_list_train.append(err_epoch)
    print("Epoch %04d err:" % i, err_epoch)
    # TODO validation

  mixed_wav_list = scipy.io.loadmat(
      "data/test_cc/mixed_wav_dir.mat")["mixed_wav_dir"]
  for mixed_wavs in mixed_wav_list:
    utt1 = mixed_wavs[0]
    utt2 = mixed_wavs[1]
    x_, y_ = utils.prepare_x_y_for_dnn(utt1, utt2, 7)
    err = _nn.test()
    print("Test_CC ERR : ", err)
