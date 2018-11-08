import os
import sys
import numpy as np
import scipy.signal
import datetime
import param
import shutil
import scipy.io
from nnet.print_ import print_,print_f


def run():
  param.val = 2
  clean_wav_speaker_set_dir = param.RAWDATA_DIR
  log_dir = "log"
  if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
  data_dir = "data"
  if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
  os.makedirs('data/train')
  os.makedirs('data/validation')
  os.makedirs('data/test_cc')
  cwl_train_file = open(data_dir+'/train/clean_wav_dir.list', 'a+')
  cwl_validation_file = open(data_dir+'/validation/clean_wav_dir.list', 'a+')
  cwl_test_cc_file = open(data_dir+'/test_cc/clean_wav_dir.list', 'a+')
  clean_wav_list_train = []
  clean_wav_list_validation = []
  clean_wav_list_test_cc = []
  speaker_list = os.listdir(clean_wav_speaker_set_dir)
  speaker_list.sort()
  for speaker_name in speaker_list:
    speaker_dir = clean_wav_speaker_set_dir+'/'+speaker_name
    if os.path.isdir(speaker_dir):
      speaker_wav_list = os.listdir(speaker_dir)
      speaker_wav_list.sort()
      for wav in speaker_wav_list[:100]:
        if wav[-4:] == ".wav":
          cwl_train_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_train.append(speaker_dir+'/'+wav)
      for wav in speaker_wav_list[260:300]:
        if wav[-4:] == ".wav":
          cwl_validation_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_validation.append(speaker_dir+'/'+wav)
      for wav in speaker_wav_list[320:]:
        if wav[-4:] == ".wav":
          cwl_test_cc_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_test_cc.append(speaker_dir+'/'+wav)

  cwl_train_file.close()
  cwl_validation_file.close()
  cwl_test_cc_file.close()
  print_('train clean: '+str(len(clean_wav_list_train)))
  print_('validation clean: '+str(len(clean_wav_list_validation)))
  print_('test_cc clean: '+str(len(clean_wav_list_test_cc)))
  print_('train mixed: '+str(len(clean_wav_list_train)*len(clean_wav_list_train)))
  print_('validation mixed: '+str(len(clean_wav_list_validation)
                                  * len(clean_wav_list_validation)))
  print_('test_cc mixed: '+str(len(clean_wav_list_test_cc)
                               * len(clean_wav_list_test_cc)))
  print_('All about: '+str(len(clean_wav_list_train)*len(clean_wav_list_train)+len(clean_wav_list_validation)
                           * len(clean_wav_list_validation)+len(clean_wav_list_test_cc)*len(clean_wav_list_test_cc)))

  data_class_dir = ['train', 'validation', 'test_cc']
  for (clean_wav_list, j) in zip((clean_wav_list_train, clean_wav_list_validation, clean_wav_list_test_cc), range(3)):
    print_(data_class_dir[j]+" data preparing...")
    print_('Current time: '+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    mixed_wav_list_file = open(
        data_dir+'/'+data_class_dir[j]+'/mixed_wav_dir.list', 'a+')
    mixed_wave_list = []
    for utt1_dir in clean_wav_list:
      for utt2_dir in clean_wav_list:
        speaker1 = utt1_dir.split('/')[-2]
        speaker2 = utt2_dir.split('/')[-2]
        if speaker1 == speaker2:
          continue
        mixed_wav_list_file.write(utt1_dir+' '+utt2_dir+'\n')
        mixed_wave_list.append([utt1_dir, utt2_dir])
    mixed_wav_list_file.close()
    scipy.io.savemat(
        data_dir+'/'+data_class_dir[j]+'/mixed_wav_dir.mat', {"mixed_wav_dir": mixed_wave_list})
  print_('Over time: '+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
