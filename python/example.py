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
    kernelTypes = 'Gauss';
    method = 'generic';
    normOut = False;
    parameterList = ['P1'];
    
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
    parameters2 = [[] for x in range(len(parameterList))]
    for indBeat,currentBeat in enumerate(singleBeatsProcessed):
        if singleBeatsProcessed[indBeat] is not None: 
            signalRec,y,optVal,normBeat = bDec.decompositionAlgorithm(currentBeat,sampleRate,numKernels=numKernels,kernelTypes=kernelTypes,method=method)
            
            """
            pwaParams = pwa.getPWAparams(currentBeat,sampleRate)
            for indParameter,currentParameter in enumerate(parameterList):
                parameters[indParameter].append(pwaParams[parameterList[indParameter]])
            """
            
            returnParameterList = pwa.calculateParameter(signalRec,currentBeat,y,optVal,sampleRate,parameterList)
            for indParameter,currentParameter in enumerate(parameterList):
                parameters2[indParameter].append(returnParameterList[indParameter])
            
            
    
    
