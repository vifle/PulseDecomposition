import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import beatDetection as bDet
import beatDecomposition as bDec
import pwaParams as pwa

def clearAll():
    plt.close('all')
    clear = lambda: os.system('cls')  # On Windows System
    clear()

if __name__ == "__main__":
    # clear workspace
    #clearAll()
    
    # settings
    sampleRate = 2000
    numKernels = 3;
    kernelTypes = 'RayleighGaussian';
    method = 'generic';
    normOut = False;
    parameterList = ['b_a','T1'];
    
    # include directories
    dataDirectory = '../data/' # directory where data folders are currently stored
    currentPatient = 'example.txt' # enter here a subject ID
    
    # load ppg data
    data = np.loadtxt(path.join(dataDirectory,currentPatient))
    ppgSignal = data[:,1];
    ppgSignalTime = data[:,0];
    
    # find beats
    ppgDetections = bDet.beatDetectionLazaro(ppgSignal,sampleRate)
    
    # segment into beats
    singleBeatsProcessed,singleBeats = bDec.beatSegmentation(ppgSignal,sampleRate,ppgDetections)
    
    # decompose beats & calculate parameters
    decompositionResults = dict()
    parameters = [[] for x in range(len(parameterList))]
    for indBeat,currentBeat in enumerate(singleBeatsProcessed):
        if singleBeatsProcessed(currentBeat):
            signalRec,y,optVal,normBeat = bDec.decompositionAlgorithm(currentBeat,sampleRate,numKernels=numKernels,kernelTypes=kernelTypes,method=method)
            
    
    
