import numpy as np
import wave
import os
#import math

def enframe(signal, nw, inc):
    signal_length=len(signal) #信号总长度
    if signal_length<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))
    pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    print(np.shape(signal))
    print(np.shape(zeros))
    pad_signal=np.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
#    win=np.tile(winfunc(nw),(nf,1))  #window窗函数，这里默认取1
#    return frames*win   #返回帧信号矩阵
    return frames
def wavread(filename):
    f = wave.open(filename,"rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#读取音频，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
    f.close()
    waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
    waveData = np.reshape(waveData,[nframes,nchannels]).T
    return waveData


filename = "utterance_test/BAC009S0908W0121.wav"
data = wavread(filename)
nw = 512
inc = 128
Frame = enframe(data[0], nw, inc)
