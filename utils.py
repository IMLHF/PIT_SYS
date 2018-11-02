import wave
import numpy as np
import scipy.io


def _manual_magnitude_spectrum_sci_stft(signal, NFFT=512, overlap=256):
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
  f, t, mag_frames = np.absolute(scipy.signal.stft(signal,
                                                   fs=16000,  # signal的采样率
                                                   window="hamming",
                                                   nperseg=NFFT,
                                                   noverlap=overlap,
                                                   nfft=NFFT,
                                                   ))
  # pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)
  return t, f, mag_frames.T


def mix_audio_and_extract_fea(file1, file2, dis_dir, fea_dir,verbose=True):
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
  mixedData = []
  for (s1, s2) in zip(waveData1, waveData2):
    s1, s2, mixed = int(s1), int(s2), 0
    # mixed=(s1+s2)//2
    if s1 < 0 and s2 < 0:
      mixed = s1+s2 - (s1 * s2 / -(pow(2, 16-1)-1))
    else:
      mixed = s1+s2 - (s1 * s2 / (pow(2, 16-1)-1))
    mixedData.append(mixed)
  mixedData = np.array(mixedData, dtype=np.int16)  # 必须指定是16位，因为写入音频时写入的是二进制数据

  name1 = file1.split('/')[-1][:-4]
  name2 = file2.split('/')[-1][:-4]
  if verbose:
    print(name1, "mix", name2)
  outfile = dis_dir+'/'+name1+'MIX'+name2+".wav"
  outwave = wave.open(outfile, 'wb')
  nchannels = 1
  sampwidth = 2  # 采样位宽，2表示16位
  framerate = 16000
  nframes = len(mixedData)
  comptype = "NONE"
  compname = "not compressed"
  outwave.setparams((nchannels, sampwidth, framerate, nframes,
                     comptype, compname))
  outwave.writeframes(mixedData)

  # 提取特征
  NFFT = 512
  overlap = 256
  mix_mag_spec = _manual_magnitude_spectrum_sci_stft(mixedData, NFFT, overlap)
  clean1_mag_spec = _manual_magnitude_spectrum_sci_stft(waveData1,NFFT,overlap)
  clean2_mag_spec = _manual_magnitude_spectrum_sci_stft(waveData2,NFFT,overlap)
  mix_log_pow_spec = np.log((1.0 / NFFT) * (mix_mag_spec ** 2))
  clean1_log_pow_spec = np.log((1.0 / NFFT) * (clean1_mag_spec ** 2))
  clean2_log_pow_spec = np.log((1.0 / NFFT) * (clean2_mag_spec ** 2))
  feature={
    "x":mix_log_pow_spec,
    "y":[clean1_log_pow_spec,clean2_log_pow_spec],
  }
  scipy.io.savemat(fea_dir+'/'+name1+"MIX"+name2+".mat",feature)


  f1.close()
  f2.close()
  outwave.close()
