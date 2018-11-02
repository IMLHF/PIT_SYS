import os
import numpy as np
import utils

clean_wav_speaker_set_dir = "utterance_test/speaker_set"
mixed_utterance_dir = "utterance_test/mixed_utterance"
fea_dir = "utterance_test/log_power_spectrum_feature"
clean_wav_list = []
for root, dirs, files in os.walk(clean_wav_speaker_set_dir):
  for file in files:
    if file[-4:] == ".wav":
      clean_wav_list.append('/'.join([root,file]))

print(clean_wav_list[0])

for utt1_dir in clean_wav_list:
  for utt2_dir in clean_wav_list:
    speaker1=utt1_dir.split('/')[-2]
    speaker2=utt2_dir.split('/')[-2]
    if speaker1==speaker2:
      continue
    utils.mix_audio_and_extract_fea(utt1_dir,utt2_dir,mixed_utterance_dir,fea_dir,verbose=True)

