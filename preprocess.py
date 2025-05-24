import numpy as np
import pandas as pd
import os
import torch
import pdb
import mat73, h5py
import neo.io

def butter_bandpass_filter(data, fmin, fmax, fs, order=1):
    """
    Bandpass filtering.
    """
    from scipy.signal import butter, filtfilt
    nyquist_freq = 0.5 * fs
    low = fmin / nyquist_freq
    high = fmax / nyquist_freq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def filter_rawdata(sig,fs,fmin=300,fmax=5000,ts=4e-4):
    """
    Perform bandpass filtering of raw data.
    Subtract moving average signal.
    """
    filtered_data = butter_bandpass_filter(sig,fmin,fmax,fs)
    
    ### Subtract moving average signal
    window = 2*(int(ts*fs)//2) + 1
    avg_data = np.convolve(filtered_data, np.ones(window)/window,mode='valid')
    avgshifted_data = filtered_data[(window//2):(-window//2+1)] - avg_data

    return avgshifted_data
    
def measure_noise(sig, R=100, K=5):
    """
    Measures noise by sampling regions from the entire signal
    over several iterations. 
    """
    R = 50
    noise_level = []
    N = sig.shape[0]

    for i in range(K):
        ranges = np.sort(np.random.randint(0,N,2*R))
        minVal, maxVal = [], []
        for i in range(R):
            minVal.append(sig[ranges[i]:ranges[i+1]].min())
            maxVal.append(sig[ranges[i]:ranges[i+1]].max())
        minVal = np.abs(np.median(np.array(minVal)))
        maxVal = np.abs(np.median(np.array(maxVal)))
        if minVal > maxVal:
            noise = 1.4*minVal
        else:
            noise = 1.4*maxVal
        noise_level.append(noise)

    noise = np.mean(noise_level)
    print("Noise value: %.4f"%noise)
    return noise

def denoise(sig, noise):
    """ Threshold signal to saturate at noise levels"""
    sig[sig > noise] = noise
    sig[sig < -noise] = -noise
    
    return sig

def event_detector(sig, thresh, pOff, nOff):
    """
    Simple event detector that folds the signal into 2d array
    of shape (-1,h,h) for faster processing
    """
    h = (pOff + nOff)
    D = len(sig) - (len(sig)%h**2)
    sig = torch.Tensor(sig[:D])
    data = sig.reshape(-1,h,h)    ### fold data into batch of 2d
    print("Shape of folded data: ",data.shape)
    
    peak_data = (data > thresh) * data              ### Get peaks in each row
    peaks = peak_data.argmax(dim=2)     
    
    ### Simple event detection

    events = []
    eventIdx = []
    for k in range(len(data)-1):
        idx = np.where(peaks[k] > 0)[0]
        for i in idx:
            t = data[k][i].view(-1) 
            j = np.argmax(t).item()
            jGlob = j+i*h+k*(h**2)

            ### Check that applies only for first event
            jGlob = nOff if jGlob < nOff else jGlob

            tmp = sig[jGlob-nOff:jGlob+pOff].unsqueeze(0)

            ### Check if max is actually max after padding
            ### from neighbouring events
            if np.argmax(tmp).item() == nOff:
                events.append(tmp)
                eventIdx.append(jGlob)


    ### Make torch dataset friendly tensor
    events = torch.cat(events,dim=0).unsqueeze(1).unsqueeze(1)
    ### Recenter individual events to 0.5 level
    events = (events - events.mean(-1,keepdim=True)) + 0.5
    E = len(eventIdx)
    print("Found %d events at %.2f threshold!" %(E,thresh))
    
    return sig, events, torch.LongTensor(eventIdx)

def thresholdEvents(events, thresh):
    posEvents = torch.where(events.max(axis=-1)[0].squeeze() > thresh)[0]
    return posEvents
 
def preprocess(data_file, cutoff, pOff, nOff, denoise_flag):
    """
    Detect all events above cutoff
    Further threshold events above threshold for analysis
    """
    ext = data_file.split('.')[-1]
    fs = None
    if ext == 'tev':  ## Assuming TDT data
        reader = neo.io.TdtIO(data_file)
        idx = np.array([reader.header['signal_streams'][i][0] for i in range(len(reader.header['signal_streams']))])

        sigIdx = np.where(idx == "b'Wav1'")[0][0]
        print("using %d analog channel"%sigIdx)
        data = (reader.read_segment(time_slice=None)).analogsignals[sigIdx]
        fs = float(data.sampling_rate)
        sig = np.asarray(data.data).reshape(-1)
        sig = filter_rawdata(sig,fs=fs) 
    elif ext == 'SMR':
        reader = neo.io.Spike2IO(data_file)
        idx = np.array([reader.header['signal_channels'][i][0] for i in range(len(reader.header['signal_channels']))])

        sigIdx = np.where(idx == "Wav1")[0][0]
        print("using %d analog channel"%sigIdx)
        data = (reader.read_segment(time_slice=None)).analogsignals[-1]
        fs = float(data.sampling_rate)
        sig = np.asarray(data.data).reshape(-1)
        sig = filter_rawdata(sig,fs=fs) 
    elif ext == 'dat':
        sig = np.fromfile(data_file,dtype=np.float32)
    elif ext == 'h5':
        data = pd.read_hdf(data_file)
        sig = data.Potential.values.copy()
    elif ext == 'txt':
        sig = np.loadtxt(data_file,dtype=np.float32,skiprows=1,delimiter='\t')[:,1]
    elif ext == 'mat':
        sig = mat73.loadmat(data_file)
        sig = sig[list(sig.keys())[0]]['values']
    elif ext == 'npy':
        data = np.load(data_file)
        fs = 24000
        sig = np.asarray(data).reshape(-1)
        sig = filter_rawdata(sig,fs=fs) 
    else:
        print('Unknown file format '+ext+' . Aborting!')
        os.exit()
#    pdb.set_trace()
#    sig = sig[50000:350000]
    N = sig.shape[0]

    print("Basic data statistics:") 
    print("N: %d, Max: %.4f, Min: %.4f"%(N,sig.max(),sig.min()))
    if denoise_flag:
        print("Denoising...")
        noise = measure_noise(sig)                  ### measure noise level
        sig = denoise(sig, noise)                   ### remove noise from sig.
    else:
        print("Proceeding without denoising step...")

    sig = (sig-sig.min())/(sig.max()-sig.min()) ### Normalise data to [0,1]

    sig, events, eventIdx = event_detector(sig, cutoff, pOff, nOff)

    return sig, events, eventIdx, torch.arange(len(eventIdx)), fs
