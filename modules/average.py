from datetime import time
import numpy as np

def equivalent_level(array):
    return 10*np.log10(np.mean(np.power(np.full(len(array), 10), array/10))) 

def sliding_mean(time, values, win=3600, step=0):

    interval = (time[1] - time[0]).seconds
    
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
        smean[i] = equivalent_level(values[int(i*step):int(i*step+win)])
        dtime.append(time[int(i*step+win/2)])

    return dtime, smean

class Average:
    def __init__(self, df, levelind):
        self.df = df
        self.levelind = levelind-1
        # self.interval = (df.iloc[1, timeind] - df.iloc[0, timeind]).seconds

    def daily(self, hour1, hour2, *args, win=3600, step=0):

        if step == 0:
            step = win
        
        NLim = ((hour2-hour1)%24*3600)//step + 1

        dailymean = np.zeros(NLim)
        dailytime = []

        for i in range(0, NLim):
            t = hour1*3600 + i*step + win//2
            t1 = hour1*3600 + i*step
            t2 = hour1*3600 + i*step + win

            if not args:
                temp = self.df
            else:
                temp = args[0]

            temp = temp.between_time(time(hour=(t1//3600)%24, minute=(t1%3600)//60, second=(t1%3600)%60), 
                                            time(hour=(t2//3600)%24, minute=(t2%3600)//60, second=(t2%3600)%60))

            dailymean[i] = equivalent_level(temp.iloc[:, self.levelind])
            dailytime.append(time(hour=(t//3600)%24, minute=(t%3600)//60, second=(t%3600)%60))

        return dailytime, dailymean
    
    def weekly(self, hour1, hour2, day1, day2, win=3600, step=0):

        week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day1 = day1.lower()
        day2 = day2.lower()
        
        if (day1 not in week) or (day2 not in week):
            raise ValueError("Arguments day1 and day2 must be a day of the week.")
        d1 = week.index(day1)
        d2 = week.index(day2)

        if d1 <= d2:
            temp = self.df.loc[(self.df.index.dayofweek >= d1) & (self.df.index.dayofweek >= d2)]
        else:
            temp = self.df.loc[(self.df.index.dayofweek <= d1) & (self.df.index.dayofweek >= d2)]

        weeklytime, weeklymean = self.daily(hour1, hour2, temp, win=win, step=step)
            
        return weeklytime, weeklymean






