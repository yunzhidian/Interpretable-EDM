import random
import numpy as np
import pandas as pd
import math
from nptdms import TdmsFile
import os
from scipy.signal import find_peaks

# train/val dataset
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

pf= 1e5
sr= 20e6
pn= 1000

def _DataLoader(file_dir_name, start_seg=None, numberOfSeg=None):
    """
    Function to load tdms data from [file_dir_name] file
    Return voltage and current data, the first axis of which corresponds to the signal segment, and
    the second of which corresponds to the number of segments 
    """
    lengthOfPulse=(int)(sr/pf) # length of one pulse (200)
    lengthOfSeg= (int)(sr/pf)*pn # length per segment (1000*200)

    tdms_file = TdmsFile(file_dir_name)
    data= tdms_file.as_dataframe()
    #  -----------extract the pulse voltage and current---------
    vol= data.iloc[:, 0].to_numpy()/100
    cur= data.iloc[:, 1].to_numpy()/100
    del tdms_file, data
    # find the peak to determine the signal starting position
    if start_seg==None:
        peaks_start, _ = find_peaks(cur[:(int)(len(cur)/10)], height=10)
        start_seg= (int)(peaks_start[0] - lengthOfPulse/1.5)
    
    # print(f'start of segment signal: {start_seg}')
    vol_pick=  np.float32(vol[start_seg:])
    cur_pick=  np.float32(cur[start_seg:])
    nb_samples= len(vol_pick)
    del vol, cur

    if numberOfSeg==None:
        # how many segments can be drawn from the recorded signal period
        numberOfSeg= math.floor(nb_samples/lengthOfSeg)
    numberOfPulses= numberOfSeg*pn
    print(f'Good! we now have {numberOfPulses} number of pulses!')
    vol_segs= np.empty([numberOfPulses,lengthOfPulse], dtype=np.float32)
    cur_segs= np.empty([numberOfPulses,lengthOfPulse], dtype=np.float32)

    # segment the signals
    for i in range(numberOfPulses):
        vol_segs[i, :] = vol_pick[i*lengthOfPulse:(i+1)*lengthOfPulse]
        cur_segs[i, :] = cur_pick[i*lengthOfPulse:(i+1)*lengthOfPulse]

    return vol_segs, cur_segs


def data_loading(file_data, file_label, start_seg, nb_segs=100):
    """
    Balance the dataset to cut off some IC and NC pulses
    nb_segs: how many segments to extract, not more than the actutal values
    """
    # load the pulse data
    vol, cur= _DataLoader(file_data, start_seg=start_seg, numberOfSeg=nb_segs)
    # load the label data
    print(f'We have retrieved {nb_segs*pn} pulses!')
    label_raw=pd.read_csv(file_label, header=None, nrows=nb_segs*pn).values
    return vol, cur, label_raw


def data_cutoff(vol, cur, label_raw, number_to_cutoff=None):
    """
    Sometimes we need to cut off some OC or ND data for a balanced dataset
    """
    if number_to_cutoff is not None:
        np.random.seed(666)

        for i, nb in enumerate(number_to_cutoff):
            if i == 0:
                index= np.where(label_raw==0)[0] # OC=0
            elif i==1:
                index= np.where(label_raw==3)[0] # ND=3
            elif i==2:
                index= np.where(label_raw==1)[0] # SC=1
            elif i==3:
                index= np.where(label_raw==2)[0] # Arc=2
            elif i==4:
                index= np.where(label_raw==4)[0] # WD=4
            elif i==5:
                index= np.where(label_raw==5)[0] # DD=4
            # replace=False means no repeated selection!
            out_index= np.random.choice(index, nb, replace=False)
            label_raw= np.delete(label_raw, out_index, axis=0)
            vol= np.delete(vol, out_index, axis=0)
            cur= np.delete(cur, out_index, axis=0)
    return vol, cur, label_raw


def train_data_prepare(vol, cur, label_raw,
                       vol_max=140, vol_min=-10, cur_max=30, cur_min=0):
    """
    Bundle the provided data (voltage, current and labels) and 
    split them into training and validation dataset
    """
    vol = (vol - vol_min) / (vol_max - vol_min)
    cur = (cur - cur_min) / (cur_max - cur_min)
    # stack voltage and current signals depth-wise (along the third axis direction)
    dat_raw= np.dstack([vol, cur])
    # label_raw=pd.read_csv(file_label, header=None, nrows=nb_segs*pn).values
    # n_classes= 6
    n_classes = len(np.unique(label_raw))
    label_raw= to_categorical(label_raw, num_classes=n_classes, dtype ="uint8")

    # split the dataset: train, validation, test
    all_indices= list(range(dat_raw.shape[0]))
    train_ind, val_ind = train_test_split(all_indices, test_size=0.2, random_state=1, stratify=label_raw)
    X_train= tf.cast(dat_raw[train_ind, :, :], tf.float32)
    X_val= tf.cast(dat_raw[val_ind, :, :], tf.float32)
    y_train= tf.convert_to_tensor(label_raw[train_ind,:])
    y_val= tf.convert_to_tensor(label_raw[val_ind,:])

    n_features= X_train.shape[-1]
    n_timesteps=X_train.shape[-2]

    print(f'feature dimension: {n_features}')
    print(f'time steps: {n_timesteps}')
    print(f'number of classes: {n_classes}')
    print(f'train size: {X_train.shape[0]}; validation size: {X_val.shape[0]}')
    return X_train, y_train, X_val, y_val, n_features, n_timesteps, n_classes


def test_data_prepare(vol, cur, label_raw, n_classes=6, 
                      ifsample=False, nb_of_samples=None,
                      vol_max=110, vol_min=-10, cur_max=30, cur_min=0):
    """
    Bundle the provided data (voltage, current and labels) for testing purpose 
    """
    if ifsample and (nb_of_samples is not None):
        np.random.seed(666)
        print(f'We take {nb_of_samples} samples for test!')
        index= np.random.choice(np.arange(len(vol)), nb_of_samples, replace=False)
        vol= vol[index,...]
        cur= cur[index,...]
        label_raw= label_raw[index,...]

    vol = (vol - vol_min) / (vol_max - vol_min)
    cur = (cur - cur_min) / (cur_max - cur_min)
    # stack voltage and current signals depth-wise (along the third axis direction)
    dat_raw= np.dstack([vol, cur])
    X_test= tf.convert_to_tensor(dat_raw)
    y_test= to_categorical(label_raw, num_classes=n_classes, dtype ="uint8")

    return X_test, y_test