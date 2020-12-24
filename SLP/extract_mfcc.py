# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:00:49 2019

@author: PKU
"""

import librosa

url = 'records/digit_0/1_0.wav'         
y, sr = librosa.load(url,sr=16000)          #sr:采样率
mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=39)

