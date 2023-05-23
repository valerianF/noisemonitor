import pandas as pd
import numpy as np

def power_mean(array):
    return 10*np.log10(np.mean(np.power(np.full(len(array), 10), array/10))) 

def sliding_mean(time, values, win=3600, step=0):

    interval = (time[1] - time[0]).seconds

    out = []
    
    step = step // interval
    win = win // interval

    N=len(values)

    if step == 0:
        nLim = max(N // win, 1)
        step = win
    else:
        nLim = max((N-win) // step, 1)

    smean = np.zeros(nLim) 
    dtime = []

    for i in range(0, nLim):
        smean[i] = power_mean(values[int(i*step):int(i*step+win)])
        dtime.append(time[int(i*step+win/2)])


    return dtime, smean
