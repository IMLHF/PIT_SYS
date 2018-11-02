import numpy as np
import scipy
import scipy.io
import scipy.signal
import wave
from PIL import Image
import time
import matplotlib
import matplotlib.pyplot as plt


class Test(object):
  def wav_read_test(self):
    f = wave.open("utterance_test/BAC009S0908W0121.wav", 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)
    waveData = np.fromstring(strData, dtype=np.int16)
    # waveData = waveData*1.0/(max(abs(waveData)))
    time = np.arange(0, nframes)*(1.0 / framerate)
    plt.plot(time, waveData)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Single channel wavedata")
    plt.grid(True)
    plt.show()

  def load_mat_test(self):
    mat = scipy.io.loadmat("noise92/white.mat")
    print(mat.keys())

  def wav_write_test(self):
    mat = scipy.io.loadmat("noise92/white.mat")
    outData = mat['white']
    outfile = 'noise92/white.wav'
    outwave = wave.open(outfile, 'wb')
    nchannels = 1
    sampwidth = 2  # 采样位宽，2表示16位
    framerate = 16000
    nframes = len(outData)
    comptype = "NONE"
    compname = "not compressed"
    outwave.setparams((nchannels, sampwidth, framerate, nframes,
                       comptype, compname))
    outwave.writeframes(outData)

  def normalization_mix_audio_test(self):
    # noise_array=scipy.io.loadmat("noise92/white.mat")['white'].flatten()
    f1 = wave.open("utterance_test/BAC009S0908W0121.wav", 'rb')
    f2 = wave.open("utterance_test/BAC009S0908W0122.wav", 'rb')
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
    # 必须指定是16位，因为写入音频时写入的是二进制数据
    mixedData = np.array(mixedData, dtype=np.int16)

    outfile = 'test/mixed.wav'
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

  def manual_magnitude_spectrum_np_fft_test(self, signal, NFFT=512, overlap=256):
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

  def sci_spectrogram_test(self, signal, NFFT=512, overlap=256):
    f, t, spec = scipy.signal.spectrogram(signal,
                                          fs=16000,  # signal的采样率
                                          window="hamming",
                                          nperseg=NFFT,
                                          noverlap=overlap,
                                          nfft=NFFT,
                                          mode="magnitude",
                                          )
    return t, f, spec.T

  def manual_magnitude_spectrum_sci_stft_test(self, signal, NFFT=512, overlap=256):
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

  def plt_specgram_test(self, signal, NFFT=512, overlap=256):
    spec, f, t, _ = plt.specgram(signal,
                                 Fs=16000,  # signal的采样率
                                 window=np.hamming(NFFT),
                                 NFFT=NFFT,
                                 mode="magnitude",
                                 noverlap=overlap,
                                 )
    # plt.cla() # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。
    # plt.clf() # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot。
    plt.close() # 关闭 window，如果没有指定，则指当前 window。
    return t, f, spec.T

  def show_spectrum(self, mat, name, t, f):
    print(np.max(mat))
    print(np.shape(mat))
    # mat = np.log10(mat)

    img = Image.fromarray(255-np.array(255.0/np.max(mat)*mat, dtype=np.int16))
    img = img.convert('L')
    img.save('test/'+name+'.jpg')

    # plt.imshow(mat)
    # plt.tight_layout()
    # plt.show()
    plt.pcolormesh(f, t, np.log10(mat),)
    plt.title('STFT Magnitude')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Time [sec]')
    plt.savefig("test/fig_"+name+'.jpg')
    # plt.show()
    plt.close()

  def spectrum_test(self):
    f = wave.open("utterance_test/BAC009S0908W0121.wav", 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)
    waveData = np.fromstring(strData, dtype=np.int16)
    self.show_spectrum(
        self.manual_magnitude_spectrum_np_fft_test(waveData, 512, 256),
        "manual_magnitude_spectrum_np_fft",
        np.arange(0, 483),
        np.arange(0, 257)
    )
    t, f, spec = self.sci_spectrogram_test(waveData, 512, 256)
    self.show_spectrum(
        spec,
        "sci_spectrogram_test",
        t,
        f,
    )
    t, f, spec = self.manual_magnitude_spectrum_sci_stft_test(
        waveData, 512, 256)
    self.show_spectrum(
        spec,
        "manual_magnitude_spectrum_sci_stft",
        t,
        f,
    )
    t, f, spec = self.plt_specgram_test(
        waveData, 512, 256)
    self.show_spectrum(
        spec,
        "plt_specgram_test",
        t,
        f,
    )


if __name__ == "__main__":
  test = Test()
  # test.wav_read_test()
  # test.load_mat_test()
  # test.wav_write_test()
  # test.normalization_mix_audio_test()
  test.spectrum_test()
