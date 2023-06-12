import math
import numpy as np
import scipy.signal as sig

def smoothDiff(n):
    # for real 1d array, the 1 in shape definition can/should be omitted
    if(n>=2 and math.floor(n)==math.ceil(n)):
        if(n%2==1):
            m = np.fix((n-1)/2)
            m = m.astype(int)
            h = np.concatenate((-np.ones((1,m)),np.zeros((1)),np.ones((1,m))),axis=None)
            h = h/m/(m+1)
        else:
            m = np.fix(n/2)
            m = m.astype(int)
            h = np.concatenate((-np.ones((1,m)),np.zeros((1)),np.ones((1,m))),axis=None)
            h = h/m/(m+1)
    else:
        raise Exception('The input parameter (n) should be a positive integer larger no less than 2.')
    return h

def beatDetectionLazaro(ppg,fs,DEBUG=False):   
    # smooth differentiator
    firOrder = round(fs/3)
    b = smoothDiff(firOrder)
    a = np.concatenate((np.ones((1)),np.zeros((len(b)-1))))
    df= (b,a)
    # NOTE: group delay in matlab calculated differently though python way is implemented
    # 12.06.2023: group_delay seems to have changed
    _,gd = sig.group_delay(df)
    delay = round(np.mean(gd[1:])) # changed this from gd to gd[1:] as first sample in gd is unreasonable
    deriv = sig.lfilter(-b,a,np.concatenate((ppg,np.zeros((1,delay))),axis=None))
    deriv = deriv[delay:]
    
    # detections in derivative
    ind,_ = sig.find_peaks(deriv,distance=round(fs*60/200))
    slope = deriv[ind]
    
    # create adaptive threshold
    thres = np.median(slope)*0.3 * np.ones((len(ppg),))#dimension of np ones?
    medianRR_samples = round(np.median(np.diff(ind[slope>thres[ind]])))
    nBeatExpect = round(medianRR_samples*0.7)
    nRefrac = round(0.15*fs)
    
    # loop through detections
    for count,index in enumerate(ind,start=0):
        if(slope[count] >= thres[index]):
            if(index+nRefrac < len(ppg)):
                thres[index:index+nRefrac+1] = slope[count]
                if(index + nRefrac + nBeatExpect < len(ppg)):
                    thres[index+nRefrac+1:index+nRefrac+nBeatExpect+1] = slope[count] - np.dot(np.arange(1,nBeatExpect+1),(slope[count]*0.8/nBeatExpect))
                    thres[index+nRefrac+nBeatExpect:] = slope[count]*0.2
                else:
                    thres[index:] = slope[count]
            else:
                thres[index:] = slope[count]
    ind = np.delete(ind,slope < thres[ind])
    return ind