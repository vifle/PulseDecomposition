import math
import numpy as np
from scipy import signal as sig
from scipy import interpolate as interp
from scipy import optimize as opt
import helperFuns

def normalize(ppgBeat):
    ppgBeat = (ppgBeat - min(ppgBeat))/(max(ppgBeat) - min(ppgBeat))
    return ppgBeat

def getNormalizationFactors(ppgBeat):
    # linear transfomation y = mx + n
    m = 1/(max(ppgBeat) - min(ppgBeat))
    n = min(ppgBeat)/(max(ppgBeat) - min(ppgBeat))
    return m,n

def beatSegmentation(ppg,sampFreq,beatIndices,DEBUG=None):
    # define segments
    segmentLength = np.diff(beatIndices)
    segmentLength = int(np.ceil(np.median(segmentLength)))
    beatInterval_before = round(0.4*segmentLength)
    beatInterval_after = segmentLength
    # first beat could be incomplete
    if(beatInterval_before > beatIndices[0]):
        try:
            minsBefore,_ = sig.find_peaks(-ppg[0:beatIndices[0]]);
        except:
            minsBefore = [];
        if(minsBefore.any()):
            firstBeat = ppg[0:beatIndices[0]+beatInterval_after]
            beatIndices = np.delete(beatIndices,0)
            insertBeat = True
        else:
            firstBeat = None
            beatIndices = np.delete(beatIndices,0)
            insertBeat = True # TODO: why insert None beat? I do this in matlab as well...seems dumb --> in matlab this is okay, as i check if single beats are none
            # could either also return ind where first ind is deleted so that number of beats and ind match
            # or could check in decomposition if ppg is None and if so give None for all outputs
            # for now go with second option
            # TODO: what about last detection...could this be too short?
    else:
        insertBeat = False
    # cut out single beats
    singleBeats = []
    for index,value in enumerate(beatIndices, start=0):
        singleBeats.append(ppg[value-beatInterval_before:value+beatInterval_after])
    if(insertBeat):
        singleBeats.insert(0, firstBeat)
    singleBeatsProcessed = []
    ## process beats
    for beatNumber,currentBeat in enumerate(singleBeats):
        # cut from min to min
        if(currentBeat is None):
            singleBeatsProcessed.append(currentBeat)
            continue
        startingSegment = currentBeat[0:beatInterval_before]
        minimaIndices,_ = sig.find_peaks(-startingSegment)
        if(minimaIndices.any()):
            beatStartIndex = minimaIndices[-1]
        else:
            beatStartIndex = 0
        endingSegment = currentBeat[beatInterval_before:]
        minimaIndices,_ = sig.find_peaks(-endingSegment)
        if(minimaIndices.any()):
            beatStopIndex = beatInterval_before + minimaIndices[np.argmin(endingSegment[minimaIndices])]
            if (beatStopIndex - beatStartIndex) < (0.4*sampFreq):
                beatStopIndex = len(currentBeat)-1
        else:
            beatStopIndex = len(currentBeat)-1
        # detrend beat
        linTrend = interp.interp1d(np.array([beatStartIndex,beatStopIndex]), np.array([currentBeat[beatStartIndex],currentBeat[beatStopIndex]]))
        trendData = linTrend(np.arange(beatStartIndex,beatStopIndex+1))
        currentBeat = currentBeat[beatStartIndex:beatStopIndex+1] - trendData
        singleBeatsProcessed.append(currentBeat)
    return singleBeatsProcessed, singleBeats

def decompositionAlgorithm(ppgBeat,sampFreq,numKernels=2,kernelTypes='GammaGauss',method='generic',initialValues=None,normalizeInput=False,noOpt=False):
    prevOverflow = False
    if ppgBeat is None:
        signalRec = y = optVal = ppgBeat = None
        return signalRec,y,optVal,ppgBeat
    # produce time axis for ppgBeat
    time = np.linspace(0,len(ppgBeat)/sampFreq,num=len(ppgBeat))
    # normalize input
    m,n = getNormalizationFactors(ppgBeat)
    ppgBeat = normalize(ppgBeat)
    # define optimization problem
    if(initialValues is None):
        x0 = np.zeros((numKernels*3));
        if(method=='generic'):
            if(numKernels==2):
                # initial values
                """
                x0[0] = 0.8*max(ppgBeat) # amplitude (a)
                x0[1] = (2/7)*max(time) # position (my)
                x0[2] = (((2/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                x0[3] = 0.5*max(ppgBeat) # amplitude (a)
                x0[4] = (4/7)*max(time) # position (my)
                x0[5] = (((3/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                """
                x0[0] = 0.9*max(ppgBeat) # amplitude (a)
                x0[1] = (1/7)*max(time) # position (my)
                x0[2] = (((1/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                x0[3] = 0.3*max(ppgBeat) # amplitude (a)
                x0[4] = (3/7)*max(time) # position (my)
                x0[5] = (((4/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                #"""
                # boundaries
                lb = np.array([0,0,0,
                0,0,0]) # lower boundary
                ub = np.array([max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time)]) # upper boundary
                # constraints
                cons = ({'type': 'ineq', 'fun': lambda x:  x[4] - x[1]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[3]})
            if(numKernels==3):
                # initial values
                """
                x0[0] = 0.8*max(ppgBeat) # amplitude (a)
                x0[1] = (2/7)*max(time) # position (my)
                x0[2] = (((2/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                """
                x0[0] = 0.9*max(ppgBeat) # amplitude (a)
                x0[1] = (1/7)*max(time) # position (my)
                x0[2] = (((1/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                x0[3] = 0.4*max(ppgBeat) # amplitude (a)
                x0[4] = (4/7)*max(time) # position (my)
                x0[5] = (((2/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                x0[6] = 0.2*max(ppgBeat) # amplitude (a)
                x0[7] = (5/7)*max(time) # position (my)
                x0[8] = (((2/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                # boundaries
                lb = np.array([0,0,0,
                0,0,0,
                0,0,0]) # lower boundary
                ub = np.array([max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time)]) # upper boundary
                # constraints
                cons = ({'type': 'ineq', 'fun': lambda x:  x[4] - x[1]},
                        {'type': 'ineq', 'fun': lambda x:  x[7] - x[4]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[3]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[6]})
            if(numKernels==4):
                # initial values
                """
                x0[0] = 0.8*max(ppgBeat) # amplitude (a)
                x0[1] = (2/7)*max(time) # position (my)
                x0[2] = (((2/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                """
                x0[0] = 0.9*max(ppgBeat) # amplitude (a)
                x0[1] = (1/7)*max(time) # position (my)
                x0[2] = (((1/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                x0[3] = 0.4*max(ppgBeat) # amplitude (a)
                x0[4] = (3/7)*max(time) # position (my)
                x0[5] = (((2/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                x0[6] = 0.4*max(ppgBeat) # amplitude (a)
                x0[7] = (1/2)*max(time) # position (my)
                x0[8] = (((2/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                x0[9] = 0.4*max(ppgBeat) # amplitude (a)
                x0[10] = (45/70)*max(time) # position (my)
                x0[11] = (((2/7)*max(time))/(2*math.sqrt(2*math.log(2)))) # width (sigma)
                # boundaries
                lb = np.array([0,0,0,
                0,0,0,
                0,0,0,
                0,0,0]) # lower boundary
                ub = np.array([max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time)]) # upper boundary
                # constraints
                cons = ({'type': 'ineq', 'fun': lambda x:  x[4] - x[1]},
                        {'type': 'ineq', 'fun': lambda x:  x[7] - x[4]},
                        {'type': 'ineq', 'fun': lambda x:  x[10] - x[7]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[3]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[6]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[9]})
    else:
        initialValues[::3] = (initialValues[::3]*m)-n
        x0 = initialValues
        if(method=='generic'):
            if(numKernels==2):
                # boundaries
                lb = np.array([0,0,0,
                0,0,0]) # lower boundary
                ub = np.array([max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time)]) # upper boundary
                # constraints
                cons = ({'type': 'ineq', 'fun': lambda x:  x[4] - x[1]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[3]})
            if(numKernels==3):
                # boundaries
                lb = np.array([0,0,0,
                0,0,0,
                0,0,0]) # lower boundary
                ub = np.array([max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time)]) # upper boundary
                # constraints
                cons = ({'type': 'ineq', 'fun': lambda x:  x[4] - x[1]},
                        {'type': 'ineq', 'fun': lambda x:  x[7] - x[4]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[3]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[6]})
            if(numKernels==4):
                # boundaries
                lb = np.array([0,0,0,
                0,0,0,
                0,0,0,
                0,0,0]) # lower boundary
                ub = np.array([max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time),
                max(ppgBeat),max(time),max(time)]) # upper boundary
                # constraints
                cons = ({'type': 'ineq', 'fun': lambda x:  x[4] - x[1]},
                        {'type': 'ineq', 'fun': lambda x:  x[7] - x[4]},
                        {'type': 'ineq', 'fun': lambda x:  x[10] - x[7]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[3]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[6]},
                        {'type': 'ineq', 'fun': lambda x:  x[0] - x[9]})
                    
    # boundaries as tuple
    bnds = []
    for boundNum,_ in enumerate(lb,start=0):
        bnds.append((lb[boundNum],ub[boundNum]))
    bnds = tuple(bnds)
    # options
    options = {"maxiter":3000,'xtol':1e-12,}
    # optimization
    def alpha0(x):
        alpha0 = (1/(2*(x[2]**2)))*(x[1]**2+x[1]*math.sqrt(x[1]**2+4*(x[2]**2)))+1
        return alpha0
    def beta0(x):
        beta0 = (1/(2*(x[2]**2)))*(x[1]+math.sqrt(x[1]**2+4*(x[2]**2)))
        return beta0
    if prevOverflow:
        # problem occurs with beta0**alpha0 in s0 calculation
        def s0(x):
            if alpha0(x) >= 1:
                # log wird negativ
                s0 = math.exp(math.log( math.exp(math.log((1/(x[0]))* math.exp((alpha0(x)-1)*math.log((alpha0(x)-1)/beta0(x))) *math.exp(1-alpha0(x))) + alpha0(x)*math.log(beta0(x))) ) - math.log(math.gamma(alpha0(x))))
            else:
                # wurzel aus was negativem
                foo = (alpha0(x)-1)/beta0(x)
                ba = alpha0(x)-1
                s0 = (1/(x[0]))*(beta0(x)**alpha0(x))/(math.gamma(alpha0(x)))*(foo)**(ba)*math.exp(1-alpha0(x))
            return s0
        def gammaKernel0(x):
            gammaKernel0 = math.exp(math.log(beta0(x[0:3]))*alpha0(x[0:3]) - math.log(math.gamma(alpha0(x[0:3])))) *1/s0(x[0:3])*np.multiply(np.power(time,(alpha0(x[0:3])-1)),np.exp(-(beta0(x[0:3]))*time))
            return gammaKernel0
        def gammaKernel1(x):
            gammaKernel1 = math.exp(math.log(beta0(x[3:6]))*alpha0(x[3:6]) - math.log(math.gamma(alpha0(x[3:6])))) *1/s0(x[3:6])*np.multiply(np.power(time,(alpha0(x[3:6])-1)),np.exp(-(beta0(x[3:6]))*time))
            return gammaKernel1
        def gammaKernel2(x):
            gammaKernel2 = math.exp(math.log(beta0(x[6:9]))*alpha0(x[6:9]) - math.log(math.gamma(alpha0(x[6:9])))) *1/s0(x[6:9])*np.multiply(np.power(time,(alpha0(x[6:9])-1)),np.exp(-(beta0(x[6:9]))*time))
            return gammaKernel2
        def gammaKernel3(x):
            gammaKernel3 = math.exp(math.log(beta0(x[9:]))*alpha0(x[9:]) - math.log(math.gamma(alpha0(x[9:])))) *1/s0(x[9:])*np.multiply(np.power(time,(alpha0(x[9:])-1)),np.exp(-(beta0(x[9:]))*time))
            return gammaKernel3
    else:
        def s0(x):
            try:
                s0 = (1/(x[0]))*(beta0(x)**alpha0(x))/(math.gamma(alpha0(x)))*((alpha0(x)-1)/beta0(x))**(alpha0(x)-1)*math.exp(1-alpha0(x))
            except OverflowError:
                s0 = float('inf')
            return s0
        def gammaKernel0(x):
            gammaKernel0 = ((beta0(x[0:3])**alpha0(x[0:3]))/(s0(x[0:3])*math.gamma(alpha0(x[0:3]))))*np.multiply(np.power(time,(alpha0(x[0:3])-1)),np.exp(-(beta0(x[0:3]))*time))
            return gammaKernel0
        def gammaKernel1(x):
            gammaKernel1 = ((beta0(x[3:6])**alpha0(x[3:6]))/(s0(x[3:6])*math.gamma(alpha0(x[3:6]))))*np.multiply(np.power(time,(alpha0(x[3:6])-1)),np.exp(-(beta0(x[3:6]))*time))
            return gammaKernel1
        def gammaKernel2(x):
            gammaKernel2 = ((beta0(x[6:9])**alpha0(x[6:9]))/(s0(x[6:9])*math.gamma(alpha0(x[6:9]))))*np.multiply(np.power(time,(alpha0(x[6:9])-1)),np.exp(-(beta0(x[6:9]))*time))
            return gammaKernel2
        def gammaKernel3(x):
            gammaKernel3 = ((beta0(x[9:])**alpha0(x[9:]))/(s0(x[9:])*math.gamma(alpha0(x[9:]))))*np.multiply(np.power(time,(alpha0(x[9:])-1)),np.exp(-(beta0(x[9:]))*time))
            return gammaKernel3
    def gaussKernel0(x):
        gaussKernel0 = x[0]*np.exp(-np.power((time-x[1]),2)/(2*x[2]**2))
        return gaussKernel0
    def gaussKernel1(x):
        gaussKernel1 = x[3]*np.exp(-np.power((time-x[4]),2)/(2*x[5]**2))
        return gaussKernel1
    def gaussKernel2(x):
        gaussKernel2 = x[6]*np.exp(-np.power((time-x[7]),2)/(2*x[8]**2))
        return gaussKernel2
    def gaussKernel3(x):
        gaussKernel3 = x[9]*np.exp(-np.power((time-x[10]),2)/(2*x[11]**2))
        return gaussKernel3
    def modelBeat(x,*args):
        if(args[0]=='GammaGauss'):
            if(args[1]==2):
                mod = gammaKernel0(x)+gaussKernel1(x)
                yTmp = np.array([gammaKernel0(x),gaussKernel1(x)])
            elif(args[1]==3):
                mod = gammaKernel0(x)+gaussKernel1(x)+gaussKernel2(x)
                yTmp = np.array([gammaKernel0(x),gaussKernel1(x),gaussKernel2(x)])
            elif(args[1]==4):
                mod = gammaKernel0(x)+gaussKernel1(x)+gaussKernel2(x)+gaussKernel3(x)
                yTmp = np.array([gammaKernel0(x),gaussKernel1(x),gaussKernel2(x),gaussKernel3(x)])
        elif(args[0]=='Gauss'):
            if(args[1]==2):
                mod = gaussKernel0(x)+gaussKernel1(x)
                yTmp = np.array([gaussKernel0(x),gaussKernel1(x)])
            elif(args[1]==3):
                mod = gaussKernel0(x)+gaussKernel1(x)+gaussKernel2(x)
                yTmp = np.array([gaussKernel0(x),gaussKernel1(x),gaussKernel2(x)])
            elif(args[1]==4):
                mod = gaussKernel0(x)+gaussKernel1(x)+gaussKernel2(x)+gaussKernel3(x)
                yTmp = np.array([gaussKernel0(x),gaussKernel1(x),gaussKernel2(x),gaussKernel3(x)])
        elif(args[0]=='Gamma'):
            if(args[1]==2):
                mod = gammaKernel0(x)+gammaKernel1(x)
                yTmp = np.array([gammaKernel0(x),gammaKernel1(x)])
            elif(args[1]==3):
                mod = gammaKernel0(x)+gammaKernel1(x)+gammaKernel2(x)
                yTmp = np.array([gammaKernel0(x),gammaKernel1(x),gammaKernel2(x)])
            elif(args[1]==4):
                mod = gammaKernel0(x)+gammaKernel1(x)+gammaKernel2(x)+gammaKernel3(x)
                yTmp = np.array([gammaKernel0(x),gammaKernel1(x),gammaKernel2(x),gammaKernel3(x)])
        return mod, yTmp
    def costFun(x,*args):
        model,_ = modelBeat(x,*args)
        h = sum(np.power(ppgBeat-model,2))
        return h
    try:
        res = opt.minimize(costFun,x0,args=(kernelTypes,numKernels,method),method='trust-constr',bounds=bnds,constraints=cons,options=options)
    except:
        signalRec = None
        y = None
        optVal = None
        if not normalizeInput:
            ppgBeat = (ppgBeat+n)/m
        return signalRec,y,optVal,ppgBeat
    # reconstruction
    optValTmp = res.x
    _,yTmp = modelBeat(optValTmp,kernelTypes,numKernels,method)
    # sorting waves
    peakTmp = np.zeros((np.size(yTmp,0),))
    for waveNum,_ in enumerate(yTmp,start=0):
        peakTmp[waveNum] = np.argmax(yTmp[waveNum])
        if(peakTmp[waveNum].size != 1):
            peakTmp[waveNum] = yTmp[waveNum].size # take last sample as peak if no peak exists
    sortInd = np.argsort(peakTmp)
    y = yTmp[sortInd]
    # sort optimized parameters
    optVal = np.empty((np.size(optValTmp)))
    for waveNum,_ in enumerate(yTmp,start=0):
        optVal[waveNum*3:3+waveNum*3] = optValTmp[sortInd[waveNum]*3:3+sortInd[waveNum]*3]
    # sum of kernels
    signalRec = sum(y)
    if not normalizeInput:
        # transform signalRec,y,ppgBeat back, also amplitude in optval
        signalRec = (signalRec+n)/m
        ppgBeat = (ppgBeat+n)/m
        y = (y+n)/m
        optVal[::3] = (optVal[::3]+n)/m
    return signalRec,y,optVal,ppgBeat

def getNRMSE(recBeat,refBeat):
    # input can be used
    if recBeat is not None and refBeat is not None:
        derivRef = helperFuns.deriv2(refBeat)
        derivRec = helperFuns.deriv2(recBeat)
        derivRSS = sum(np.power(derivRef-derivRec,2))
        derivStd = sum(np.power(derivRef-np.mean(derivRef),2))
        nrmse = 1 - math.sqrt(derivRSS)/math.sqrt(derivStd)
    else:
        nrmse = None
    return nrmse

def getPWDparams(signalRec,y,optVal,sampFreq):
    # if decomposition failed, give output nonetheless
    if signalRec is not None:
        # necessary calculations
        time = np.linspace(0,len(signalRec)/sampFreq,num=len(signalRec))
        numKernels = int(optVal.size/3)
        # kernel characteristics
        p = np.empty((numKernels))
        t = np.empty((numKernels))
        w = np.empty((numKernels))
        for currentKernel,_ in enumerate(p,start = 0):
            p[currentKernel] = optVal[3*currentKernel]
            t[currentKernel] = optVal[3*currentKernel+1]
            w[currentKernel] = optVal[3*currentKernel+2]
        # param combinations
        pRatio = np.empty((numKernels-1))
        tDiff = np.empty((numKernels-1))
        for currentKernel,_ in enumerate(pRatio,start = 0):
            pRatio[currentKernel] = p[currentKernel+1]/p[0]
            tDiff[currentKernel] = t[currentKernel+1] - t[0]
        riPeak = np.max(sum(y[1:]))/np.max(y[0])
        riArea = np.trapz(sum(y[1:]))/np.trapz(y[0])
        tSys = time[np.argmax(y[0])]
        tDia = np.dot(np.max(y[1:],axis=1),time[np.argmax(y[1:],axis=1)])/(sum(np.max(y[1:],axis=1)))
        tSysDia = tDia - tSys 
        pwdParams = {
            "Amplitude":p,
            "Time":t,
            "Width":w,
            "P1Px":pRatio,
            "T1Tx":tDiff,
            "RIpeak":riPeak,
            "RIarea":riArea,
            "tSysDia":tSysDia
            }
    else:
        pwdParams = {
            "Amplitude":None,
            "Time":None,
            "Width":None,
            "P1Px":None,
            "T1Tx":None,
            "RIpeak":None,
            "RIarea":None,
            "tSysDia":None
            }
    return pwdParams