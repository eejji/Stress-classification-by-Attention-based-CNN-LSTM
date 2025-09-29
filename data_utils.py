#!/usr/bin/env python
# coding: utf-8

# In[2]:


import neurokit2 as nk
import numpy as np
from scipy import signal, stats
from scipy.signal import butter, medfilt
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler

np.random.seed(7)

def shuffle_data(data, labels):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    return data[indices], labels[indices]


def shuffle_data2(data1, data2, labels):
    # data1, data2, label을 입력받아 무작위로 섞고 (인덱스 위치는 동일하게) Return
    indices = np.arange(data1.shape[0])
    np.random.shuffle(indices)
    return data1[indices], data2[indices], labels[indices]

# segmentation data
def load_BEAT_data(signal, r_peaks, left, right):
    signal_list = []
    for number in range(1,len(r_peaks)-1):
        signal_list.append(signal[r_peaks[number]-left : r_peaks[number]+right])
    return np.array(signal_list)

def load_RHYTHM_data(data, rpeak, window, right):
    rhythm_data = []
    for i in range(1,len(rpeak)-1):
        if len(data[rpeak[i] + right : rpeak[i] + window + right]) == window:
            rhythm_data.append(data[rpeak[i] + right : rpeak[i]+right + window])
            
    return np.array(rhythm_data)


# get R peak
def get_rpeak(signal, sampling_rate):
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate, show=False, method='neurokit')
    
    return rpeaks['ECG_R_Peaks']

# Denoise
def butter_highpass(ecg, cutoff, Fs, order):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, ecg)
    return y

def butter_bandstop(ecg, low, high, Fs, order):
    nyq = 0.5 * Fs
    low = low/ nyq
    high = high/ nyq
    b, a = signal.butter(order, [low, high], btype='bandstop')
    y = signal.filtfilt(b,a,ecg)
    return y

def butter_lowpass(ecg, cutoff, Fs, order):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, ecg)
    return y


# Downsampling
def Resampling(data, original_freq, new_freq):
    original_samples = len(data)
    new_samples = int(original_samples * (new_freq / original_freq))
    
    downsampled = resample(data, new_samples)
    
    return downsampled

# Normalization
def MinMax(data):
    
    scaler = MinMaxScaler()

    try:
        scaled_data = scaler.fit_transform(data)
        
    except ValueError:
        scaled_data = scaler.fit_transform(data.reshape(-1,1))
    
    return scaled_data

# outlier
def Delet_outlier(data, high_level = None, low_level = None):
    
    high = False if high_level is None else high_level
    low = False if low_level is None else low_level
    
    ex_data = pd.Series(data)
    Q1 = ex_data.quantile(0.25)
    Q3 = ex_data.quantile(0.75)
    IQR = Q3 - Q1 
    
    if high and low:
        dff = ex_data[(ex_data <= Q3+(high*IQR)) & (ex_data >= Q1-(low*IQR))]
        
    elif high and not low:
        dff = ex_data[(ex_data <=Q3+(high*IQR))]
        
    elif not high and low:
        dff = ex_data[(ex_data >= Q1-(low*IQR))]
    else :
        dff = ex_data
    
    #dff = ex_data[(ex_data <= Q3+(high_level*IQR)) & (ex_data >= Q1-(low_level*IQR))]
    
    dff = dff.reset_index(drop=True)
    
    return np.array(dff)


def load_DREAMER_class(Dreamer_data, segment, window=0):
    # Dataset
    Dreamer_data = pd.read_pickle(Dreamer_data)
    print(segment)
    # Raw data
    D_EXCITEMENT = Dreamer_data['excitement']
    D_NEUTRAL = Dreamer_data['baseline']
    D_STRESS = Dreamer_data['stress']
    
    # detect R-peak : Dreamer raw data
    D_excite_rpeak = du.get_rpeak(D_EXCITEMENT, sampling_rate)
    D_neutral_rpeak = du.get_rpeak(D_NEUTRAL, sampling_rate)
    D_stress_rpeak = du.get_rpeak(D_STRESS, sampling_rate)
    
    if segment == 'rhythm':
        # dreamer rhythm data
        excite = du.load_RHYTHM_data(D_EXCITEMENT, D_excite_rpeak, window, right)
        neutral = du.load_RHYTHM_data(D_NEUTRAL, D_neutral_rpeak, window, right)
        stress = du.load_RHYTHM_data(D_STRESS, D_stress_rpeak, window, right)
        
    else :
        # dreamer beat data
        excite = du.load_BEAT_data(D_EXCITEMENT, D_excite_rpeak, left, right)
        neutral = du.load_BEAT_data(D_NEUTRAL, D_neutral_rpeak, left, right)
        stress = du.load_BEAT_data(D_STRESS, D_stress_rpeak, left, right)

    return excite, neutral, stress

# def load_data(subjects, segment, window=0):
    
#     EXCITEMENT, NEUTRAL, STRESS =[], [], []

#     for i in subjects:
#         df = pd.read_pickle(Data_path + '/' + i + '.pkl')

#         excite_data = df['excitement'].flatten()
#         neutral_data = df['neutral'].flatten()
#         stress_data = df['stress'].flatten()

#         excite_rpeaks = du.get_rpeak(excite_data, sampling_rate)
#         neutral_rpeaks = du.get_rpeak(neutral_data, sampling_rate)
#         stress_rpeaks = du.get_rpeak(stress_data, sampling_rate)
        
#         if segment == 'rhythm':
#             EXCITEMENT.append(du.load_RHYTHM_data(excite_data, excite_rpeaks, window, right))
#             NEUTRAL.append(du.load_RHYTHM_data(neutral_data, neutral_rpeaks, window, right))
#             STRESS.append(du.load_RHYTHM_data(stress_data, stress_rpeaks, window, right))
#         else :
#             EXCITEMENT.append(du.load_BEAT_data(excite_data, excite_rpeaks, left, right))
#             NEUTRAL.append(du.load_BEAT_data(neutral_data, neutral_rpeaks, left, right))
#             STRESS.append(du.load_BEAT_data(stress_data, stress_rpeaks, left, right))
        
#     print('........' + str(window) + ' Window Data load')
#     EXCITEMENT = np.vstack(EXCITEMENT)
#     NEUTRAL = np.vstack(NEUTRAL)
#     STRESS = np.vstack(STRESS)
    
#     EXCITEMENT_label = np.zeros(EXCITEMENT.shape[0], np.int32)
#     NEUTRAL_label = np.ones(NEUTRAL.shape[0], np.int32)
#     STRESS_label = 2*np.ones(STRESS.shape[0], np.int32)
    
#     Train = np.concatenate([EXCITEMENT, NEUTRAL, STRESS])
#     Train_label = np.concatenate([EXCITEMENT_label, NEUTRAL_label, STRESS_label])
    
#     train, train_y = du.shuffle_data(Train, Train_label)
    
#     return train, train_y