import numpy as np
from scipy import signal as sig
import beatDecomposition as bDec
import helperFuns

def getSystolicPeak(beat):
    # get index of first highest maximum
    peaks,_ = sig.find_peaks(beat)
    if(peaks.size == 0):
        sysInd = None
    else:
        sysInd = peaks[np.argmax(beat[peaks])] 
    return sysInd

def getDiastolicPeak(beat):
    # get highest peak after  systolic peak
    sysInd = getSystolicPeak(beat)
    if sysInd is None:
        diaInd = None
    else:
        peaks,_ = sig.find_peaks(beat[sysInd:],height=0.15*beat[sysInd])
        if(peaks.size == 0):
            diaInd = None
        else:
            peaks = peaks + sysInd
            diaInd = peaks[np.argmax(beat[peaks])] # get index of first highest maximum
    return diaInd

def getDicroticNotch(beat):
    sysInd = getSystolicPeak(beat)
    diaInd = getDiastolicPeak(beat)
    if sysInd is None or diaInd is None:
        dicNotInd = None
    else:
        dicNotInd = np.argmin(beat[sysInd:diaInd])
        dicNotInd = dicNotInd + sysInd
    return dicNotInd

def getSysPressure(beat):
    sysInd = getSystolicPeak(beat)
    if sysInd is None:
        sp = None
    else:
        sp = beat[sysInd]
    return sp

def getDiaPressure(beat):
    dp = np.min(beat)
    return dp

def getPulsePressure(beat):
    sp = getSysPressure(beat)
    dp = getDiaPressure(beat)
    if sp is None or dp is None:
        pp = None
    else:
        pp = sp - dp
    return pp

def getMeanPressure(beat):
    mp = np.mean(beat)
    return mp

def getSysMeanPressure(beat):
    dicNotInd = getDicroticNotch(beat)
    if dicNotInd is None:
        smp = None
    else:
        smp = np.mean(beat[0:dicNotInd])
    return smp

def getDiaMeanPressure(beat):
    dicNotInd = getDicroticNotch(beat)
    if dicNotInd is None:
        dmp = None
    else:
        dmp = np.mean(beat[dicNotInd:])
    return dmp

def getEndSysPressure(beat):
    dicNotInd = getDicroticNotch(beat)
    if dicNotInd is None:
        esp = None
    else:
        esp = beat[dicNotInd]
    return esp

def getLVET(beat,sampFreq):
    # return LVET in seconds
    time = np.linspace(0,len(beat)/sampFreq,num=len(beat))
    dicNotInd = getDicroticNotch(beat)
    if dicNotInd is None:
        lvet = None
    else:
        lvet = time[dicNotInd]
    return lvet

def getDiaDur(beat,sampFreq):
    # return diastolic duration in seconds
    time = np.linspace(0,len(beat)/sampFreq,num=len(beat))
    dicNotInd = getDicroticNotch(beat)
    if dicNotInd is None:
        dd = None
    else:
        dd = time[-1] - time[dicNotInd]
    return dd

def getTimeToPeak(beat,sampFreq):
    # return time to systolic peak in seconds
    time = np.linspace(0,len(beat)/sampFreq,num=len(beat))
    sysInd = getSystolicPeak(beat)
    if sysInd is None:
        ttp = None
    else:
        ttp = time[sysInd]
    return ttp

def getFormFactor(beat):
    mp = getMeanPressure(beat)
    pp = getPulsePressure(beat)
    dp = getDiaPressure(beat)
    if pp is None or dp is None:
        ff = None
    else:
        ff = (mp - dp)/pp
    return ff

def getSPTI(beat,sampFreq):
    lvet = getLVET(beat,sampFreq)
    smp = getSysMeanPressure(beat)
    if lvet is None or smp is None:
        spti = None
    else:
        spti = lvet*smp
    return spti

def getDPTI(beat,sampFreq):
    dd = getDiaDur(beat,sampFreq)
    dmp = getDiaMeanPressure(beat)
    if dd is None or dmp is None:
        dpti = None
    else:
        dpti = dd*dmp
    return dpti

def getSEVR(beat,sampFreq):
    spti = getSPTI(beat,sampFreq)
    dpti = getDPTI(beat,sampFreq)
    if spti is None or dpti is None:
        sevr = None
    else:
        sevr = dpti/spti
    return sevr

def getRI(beat):
    sp = getSysPressure(beat)
    dia = getDiastolicPeak(beat)
    if sp is None or dia is None:
        ri = None
    else:
        ri = beat[dia]/sp    
    return ri

def getDeltaT(beat,sampFreq):
    #time = np.arange(0,len(beat)/sampFreq,1/sampFreq) # TODO: keep this for legacy for now
    time = np.linspace(0,len(beat)/sampFreq,num=len(beat))
    sysInd = getSystolicPeak(beat)
    diaInd = getDiastolicPeak(beat)
    if sysInd is None or diaInd is None:
        deltaT = None
    else:
        deltaT = time[diaInd] - time[sysInd]
    return deltaT

def getPWHA(beat,sampFreq):
    time = time = np.linspace(0,len(beat)/sampFreq,num=len(beat))
    maxAmp = np.max(beat)
    maxInd = np.argmax(beat)
    indBegin = np.where(beat>=maxAmp/2)[0][0]
    endInterval = beat[maxInd:]
    #indEnd = np.where(endInterval<=maxAmp/2)[0][0]
    indEnd = np.where(endInterval>=maxAmp/2)[0][-1]
    indEnd = indEnd + maxInd
    pwha = time[indEnd] - time[indBegin]
    return pwha

def getImpPointsSorelli(beat,sampFreq,forOpt=False,DEBUG=False):
    """
    1. search systolic reference
    """
    sys = getSystolicPeak(beat)
    """
    2. find end-diastolic perfusion through as absolute minimum (but not first point in signal)
    """
    leaveOut = 5
    diaEnd = np.argmin(beat[leaveOut:]) # will always yield last sample because of segmentation
    diaEnd = diaEnd+leaveOut # add samples to be in line with real signal input
    """
    3. refinement step for systolic detection: search for all maxima preceding 
    the original solution, whose prominence with respect to the corresponding 
    valley does not fall below 80% of the original peak amplitude
    """
    maxima,_ = sig.find_peaks(beat,prominence=0.8*beat[sys])
    maxima = np.delete(maxima,maxima>=sys)
    if maxima.size > 0:
        sys = maxima[0]
    """
    4. refinement step for diastolic detection
    not needed in this function because input signals should be beats from minimum to minimum
    """
    """
    5. calculte derivative (by 3point differentiator) (central difference)
    """
    deriv_1 = helperFuns.deriv1Central(beat)
    """
    6. filter derivative with 7point moving average
    """
    windowWidth = 7
    b = (1/windowWidth)*np.ones((windowWidth,))
    a = 1
    if beat.shape[0] <= 3*windowWidth: # avoid *** ValueError: The length of the input vector x must be greater than padlen, which is 21.
        inc = dia = None
        return sys,inc,dia
    deriv_1 = sig.filtfilt(b,a,deriv_1)
    """
    7. search for earliest neg to pos zero crossing of first derivative after 
    systolic reference. If none are detected, end-diastolic reference is selected
    """
    sign_deriv_1 = np.sign(deriv_1) 
    zeroCross = np.arange(0,len(beat))[np.append(np.diff(sign_deriv_1),0) == 2] # find sample of neg to pos change
    if zeroCross.size > 0:
        zeroCross = np.delete(zeroCross,zeroCross<=sys)
        if zeroCross.size > 0:
            cross = zeroCross[0]
        else:
            cross = diaEnd
    else:
        cross = diaEnd
    """
    8. time span between this point and the systolic peak is analyzed for the
    presence of local p'(t) maxima exceeding the average pulse slope (average
    of first derivative in this interval) in the same interval: if detected,
    the earliest of them is adopted as the incisura reference, otherwise the
    original p'(t) zero crossing is chosen
    """
    avSlope = np.mean(deriv_1[sys:cross+1])
    try:
        inc,_ = sig.find_peaks(deriv_1[sys:cross+1],height=avSlope)
    except:
        inc= np.array(())
    if inc.size > 0:
        inc = inc + sys
    else:
        if cross != diaEnd:
            inc = cross
        else:
            inc = sys
    if inc.size > 0:
        if not np.isscalar(inc):
            inc = inc[0]
    
    if DEBUG:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(beat)
        plt.plot(deriv_1)
    
    """
    additional step: estimate diastolic peak after incisura
    # TODO:
    # try finding peaks after incisura and take first one
    # if no peaks are found search for something in derivtive? Or take inc as dia peak as this is the best guess
    # there could be peaks at the end of the signal, which should not be taken as diastolic peak
    # --> peak needs to be higher than incisura
    
    # what i have now is good for optimization, but for pwa, i should not leave inc as sys,
    # but rather say that there is not inc and no dia
    # thus for class 4 beats one has to say that tSysDia cannot be calculated
    # for this case one should use decomposition for an estimate
    """
    if not forOpt:
        dia = getDiastolicPeak(beat)
        if dia is None:
            if inc == sys:
                inc = None
                dia = None
                return sys,dia,inc # return systolic, diastolic and incisura index
            dia,_ = sig.find_peaks(beat[inc:],height=beat[inc])
            dia = dia + inc
            if dia.size > 1:
                dia = dia[0]
            if dia.size == 0:
                dia = inc
            return sys,dia,inc # return systolic, diastolic and incisura index
        else:
            if inc == sys:
                inc = None
                return sys,dia,inc # return systolic, diastolic and incisura index
            else:
                return sys,dia,inc # return systolic, diastolic and incisura index
    else:
        return sys,diaEnd,inc # return systolic, diastolic and incisura index
    
def getDeltaTSorelli(beat,sampFreq):
    time = np.linspace(0,len(beat)/sampFreq,num=len(beat))
    sysInd,diaInd,_ = getImpPointsSorelli(beat,sampFreq)
    if sysInd is None or diaInd is None:
        deltaT = None
    else:
        deltaT = time[diaInd] - time[sysInd]
    return deltaT

def getBoverA(beat):
    if beat is None:
        b_a = None
        return b_a
    secondDeriv = helperFuns.deriv2(beat)
    a = max(secondDeriv)
    b = min(secondDeriv)
    aLoc = np.argwhere(secondDeriv==a)[0][0]
    bLoc = np.argwhere(secondDeriv==b)[0][0]
    if aLoc > bLoc:
        b_a = None
        return b_a
    b_a = b/a
    return b_a

def getPWAparams(beat,sampFreq):
    try:
        pwaParams = {
            "SystolicPeak":getSystolicPeak(beat),
            "DiastolicPeak":getDiastolicPeak(beat),
            "DicroticNotch":getDicroticNotch(beat),
            "SysPressure":getSysPressure(beat),
            "DiaPressure":getDiaPressure(beat),
            "PulsePressure":getPulsePressure(beat),
            "MeanPressure":getMeanPressure(beat),
            "SysMeanPressure":getSysMeanPressure(beat),
            "DiaMeanPressure":getDiaMeanPressure(beat),
            "EndSysPressure":getEndSysPressure(beat),
            "LVET":getLVET(beat,sampFreq),
            "DiaDur":getDiaDur(beat,sampFreq),
            "TimeToPeak":getTimeToPeak(beat,sampFreq),
            "FormFactor":getFormFactor(beat),
            "SPTI":getSPTI(beat,sampFreq),
            "DPTI":getDPTI(beat,sampFreq),
            "SEVR":getSEVR(beat,sampFreq),
            "RI":getRI(beat),
            "deltaT":getDeltaT(beat,sampFreq),
            "deltaTSorelli":getDeltaTSorelli(beat,sampFreq),
            "PWHA":getPWHA(beat,sampFreq),
            "b_a":getBoverA(beat)
            }
    except:
        pwaParams = {
                "SystolicPeak":None,
                "DiastolicPeak":None,
                "DicroticNotch":None,
                "SysPressure":None,
                "DiaPressure":None,
                "PulsePressure":None,
                "MeanPressure":None,
                "SysMeanPressure":None,
                "DiaMeanPressure":None,
                "EndSysPressure":None,
                "LVET":None,
                "DiaDur":None,
                "TimeToPeak":None,
                "FormFactor":None,
                "SPTI":None,
                "DPTI":None,
                "SEVR":None,
                "RI":None,
                "deltaT":None,
                "deltaTSorelli":None,
                "PWHA":None,
                "b_a":None
                }
    return pwaParams