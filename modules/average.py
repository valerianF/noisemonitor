from datetime import datetime, time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

def equivalent_level(array):
    return 10*np.log10(np.mean(np.power(np.full(len(array), 10), array/10))) 

def level_plot(df, *args):
    if isinstance(df.index[0], pd.Timestamp):
        x = df.index.to_pydatetime()
    else:
        x = [datetime.combine(datetime(10, 10, 10), i) for i in df.index]

    plt.figure(figsize=(10,8), dpi=100)

    for i in args:
        plt.plot(x, df.loc[:, i], label=i)

    if type(df.index[0]) in [pd.Timestamp, datetime]:
        plt.gcf().autofmt_xdate()
    elif type(df.index[0]) == time:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.grid(linestyle='--')
    plt.xticks(rotation = 45)
    plt.legend()
    plt.show()

    return


class Average:
    def __init__(self, df):
        self.df = df

    def daily(self, hour1, hour2, *args, win=3600, step=0):

        if step == 0:
            step = win
        
        NLim = ((hour2-hour1)%24*3600)//step + 1

        dailymean = np.zeros(NLim)
        dailyL10 = np.zeros(NLim)
        dailyL50 = np.zeros(NLim)
        dailyL90 = np.zeros(NLim)
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

            dailymean[i] = equivalent_level(temp.iloc[:, 0])
            dailyL10[i] = np.percentile(temp.iloc[:, 0], 90)
            dailyL50[i] = np.percentile(temp.iloc[:, 0], 50)
            dailyL90[i] = np.percentile(temp.iloc[:, 0], 10)
            dailytime.append(time(hour=(t//3600)%24, minute=(t%3600)//60, second=(t%3600)%60))

        dailymeandf = pd.DataFrame(index=dailytime, data={'Leq': dailymean, 
                                                          'L10': dailyL10,
                                                          'L50': dailyL50,
                                                          'L90': dailyL90})

        return dailymeandf
    
    def weekly(self, hour1, hour2, day1, day2, win=3600, step=0):

        week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day1 = day1.lower()
        day2 = day2.lower()
        
        if (day1 not in week) or (day2 not in week):
            raise ValueError("Arguments day1 and day2 must be a day of the week.")
        d1 = week.index(day1)
        d2 = week.index(day2)

        if d1 <= d2:
            temp = self.df.loc[(self.df.index.dayofweek >= d1) & (self.df.index.dayofweek <= d2)]
        else:
            temp = self.df.loc[(self.df.index.dayofweek <= d1) & (self.df.index.dayofweek >= d2)]

        weeklymeandf = self.daily(hour1, hour2, temp, win=win, step=step)
            
        return weeklymeandf
    
    def overall(self, win=3600, step=0):

        interval = (self.df.index[1] - self.df.index[0]).seconds
        
        step = step // interval
        win = win // interval

        N=len(self.df)

        if step == 0:
            NLim = max(N // win, 1)
            step = win
        else:
            NLim = max((N-win) // step, 1)

        overallmean = np.zeros(NLim) 
        overallL10 = np.zeros(NLim) 
        overallL50 = np.zeros(NLim)
        overallL90 = np.zeros(NLim)
        overalltime = []

        for i in range(0, NLim):
            overallmean[i] = equivalent_level(self.df.iloc[int(i*step):int(i*step+win), 0])
            overallL10[i] = np.percentile(self.df.iloc[int(i*step):int(i*step+win), 0], 90)
            overallL50[i] = np.percentile(self.df.iloc[int(i*step):int(i*step+win), 0], 50)
            overallL90[i] = np.percentile(self.df.iloc[int(i*step):int(i*step+win), 0], 10)
            overalltime.append(self.df.index[int(i*step+win/2)])

        meandf = pd.DataFrame(index=overalltime, data={'Leq': overallmean, 
                                                          'L10': overallL10,
                                                          'L50': overallL50,
                                                          'L90': overallL90})

        return meandf
    
    def Lden(self, day1=None, day2=None, values=False):

        if all(day is not None for day in [day1, day2]):
            week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            day1 = day1.lower()
            day2 = day2.lower()
            
            if (day1 not in week) or (day2 not in week):
                raise ValueError("Optional arguments day1 and day2 must be a day of the week.")
            d1 = week.index(day1)
            d2 = week.index(day2)

            if d1 <= d2:
                temp = self.df.loc[(self.df.index.dayofweek >= d1) & (self.df.index.dayofweek >= d2)]
            else:
                temp = self.df.loc[(self.df.index.dayofweek <= d1) & (self.df.index.dayofweek >= d2)]
        else:
            temp = self.df

        lday = equivalent_level(temp.between_time(time(hour=7), time(hour=19)).iloc[:,0])
        levening = equivalent_level(temp.between_time(time(hour=19), time(hour=23)).iloc[:,0])
        lnight = equivalent_level(temp.between_time(time(hour=23), time(hour=7)).iloc[:,0])

        lden = 10*np.log10((12*np.power(10, lday/10)
                            + 4*np.power(10, (levening+5)/10)
                            + 8*np.power(10, (lnight+10)/10))/24) 

        if values:
            return np.round(lden,2), np.round(lday,2), np.round(levening,2), np.round(lnight,2)
        return lden





