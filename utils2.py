import wave
import numpy as np
import scipy.io


def _manual_magnitude_spectrum_sci_stft(signal, NFFT=512, overlap=256):
  # print('signal', np.shape(signal))
  f, t, mag_frames = np.absolute(scipy.signal.stft(signal,
                                                   fs=16000,  # signal的采样率
                                                   window="hamming",
                                                   nperseg=NFFT,
                                                   noverlap=overlap,
                                                   nfft=NFFT,
                                                   padded=True
                                                   ))
  # pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)
  return mag_frames.T


def _manual_magnitude_spectrum_np_fft(signal, NFFT=512, overlap=256):
  segsize = NFFT  # 每帧长度
  inc = segsize-overlap
  signal_length = len(signal)
  nframes = 1 + int(np.ceil(float(np.abs(signal_length - segsize)) / inc))
  pad_length = int((nframes-1)*inc+segsize)  # 补0后的长度
  zeros = np.zeros((pad_length-signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
  pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
  indices = np.tile(np.arange(0, segsize), (nframes, 1))+np.tile(
      np.arange(0, nframes*inc, inc), (segsize, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
  indices = np.array(indices, dtype=np.int32)  # 展开overlap的帧矩阵
  frames = pad_signal[indices]  # 得到展开后帧信号矩阵
  frames *= np.hamming(segsize)  # 汉明窗
  mag_frames = np.absolute(np.fft.rfft(frames,
                                       NFFT,
                                       axis=1,
                                       ))
  # pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)
  return mag_frames

def __extract_norm_log_mag_spec(mag_spec):
  LOG_NORM_MAX = 5
  LOG_NORM_MIN = -3
  # 归一化的幅度谱对数
  log_mag_spec = np.log10(mag_spec)
  # log_power_spectrum_normalization
  log_mag_spec[log_mag_spec > LOG_NORM_MAX] = LOG_NORM_MAX
  log_mag_spec[log_mag_spec < LOG_NORM_MIN] = LOG_NORM_MIN
  log_mag_spec += np.abs(LOG_NORM_MIN)
  log_mag_spec /= LOG_NORM_MAX
  return log_mag_spec

def _mix_audio_and_extract_feature(file1, file2, verbose=True):
  # 混合语音
  f1 = wave.open(file1, 'rb')
  f2 = wave.open(file2, 'rb')
  waveData1 = np.fromstring(f1.readframes(f1.getnframes()),
                            dtype=np.int16)
  waveData2 = np.fromstring(f2.readframes(f2.getnframes()),
                            dtype=np.int16)
  if len(waveData1) < len(waveData2):
    waveData1, waveData2 = waveData2, waveData1
  # print(np.shape(waveData1))
  gap = len(waveData1)-len(waveData2)
  waveData2 = np.concatenate(
      (waveData2, np.random.randint(-400, 400, size=(gap,))))
  mixedData = ((waveData1.astype(np.float32) +
                waveData2.astype(np.float32))/2.0).astype(np.int16)
  mixedData = np.array(mixedData, dtype=np.int16)  # 必须指定是16位，因为写入音频时写入的是二进制数据

  name1 = file1.split('/')[-1][:-4]
  name2 = file2.split('/')[-1][:-4]
  if verbose:
    print(name1, "mix", name2)
  # # 混合语音写入文件
  # outfile = dis_dir+'/'+name1+'MIX'+name2+".wav"
  # outwave = wave.open(outfile, 'wb')
  # nchannels = 1
  # sampwidth = 2  # 采样位宽，2表示16位
  # framerate = 16000
  # nframes = len(mixedData)
  # comptype = "NONE"
  # compname = "not compressed"
  # outwave.setparams((nchannels, sampwidth, framerate, nframes,
  #                    comptype, compname))
  # outwave.writeframes(mixedData)

  # 提取特征
  NFFT = 512
  overlap = 256

  clean1_mag_spec = _manual_magnitude_spectrum_sci_stft(
      waveData1, NFFT, overlap)
  clean2_mag_spec = _manual_magnitude_spectrum_sci_stft(
      waveData2, NFFT, overlap)
  mix_mag_spec = _manual_magnitude_spectrum_sci_stft(
      mixedData, NFFT, overlap)
  clean1_mag_spec =__extract_norm_log_mag_spec(clean1_mag_spec)  # fix normalization
  clean2_mag_spec =__extract_norm_log_mag_spec(clean2_mag_spec)
  mix_mag_spec =__extract_norm_log_mag_spec(mix_mag_spec)
  feature_dict = {
      "x": mix_mag_spec,
      "y": [clean1_mag_spec, clean2_mag_spec],
  }
  # scipy.io.savemat(fea_dir+'/'+name1+"MIX"+name2+".mat",feature_dict)

  f1.close()
  f2.close()
  # outwave.close()
  return feature_dict


# 时间维度上补0，将时间维度上的重叠展开，然后将特征矩阵降维为向量; unfold_width,展开的宽度，即网络输入的维度，必须是奇数
def _feature_pad_unfold_flatten(feature, unfold_width):
  feature = np.array(feature)  # [none,257]
  feature = np.concatenate([np.zeros([unfold_width//2, np.shape(feature)[1]]),
                            feature,
                            np.zeros([unfold_width//2, np.shape(feature)[1]])
                            ],
                           axis=0)
  unfolded_feature = []  # [none+,257]
  for i in range(len(feature)-unfold_width+1):
    unfolded_feature.extend(feature[i:i+unfold_width])
  unfolded_feature = np.reshape(
      unfolded_feature, [-1, unfold_width*np.shape(feature)[1]])
  return unfolded_feature


def prepare_x_y_for_dnn(utt1file, utt2file, unfold_width):  # 归一化的log_power_spectrum
  feature_dict = _mix_audio_and_extract_feature(
      utt1file, utt2file, verbose=False)
  x_unfolded = _feature_pad_unfold_flatten(
      feature_dict["x"], unfold_width)  # [none,257]
  y1 = feature_dict["y"][0]  # [none-,257]
  y2 = feature_dict["y"][1]  # [none-,257]
  y = np.concatenate(
      [y1, y2], axis=1)  # [none,257*7]
  return x_unfolded, y


def prepare_x_for_rbm(utt1file, utt2file, unfold_width):
  feature_dict = _mix_audio_and_extract_feature(
      utt1file, utt2file, verbose=False)
  x_unfolded = _feature_pad_unfold_flatten(
      feature_dict["x"], unfold_width)  # [none,257]
  return x_unfolded
