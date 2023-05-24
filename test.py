from modules.load import *
from modules.average import *

path = "D:\\Files\\OneDrive - McGill University\\Work\\Fleurs de Macadam - Musiroy\\Acou_Data\\data_NS\\2020\\May\\Capteur 962_2020_04_28__15h00m00sHHMMSSms.xls"

df = load_data([path], header=1, type='NoiseSentry')

# time, mean = sliding_mean(df.iloc[:,0], df.iloc[:,2])

Av = Average(df, 2)

time, mean = Av.daily(23, 7, win=3600, step=1200)

time2, mean2 = Av.weekly(0, 23, 'monday', 'friday', win=3600, step=1200)
time3, mean3 = Av.weekly(0, 23, 'saturday', 'sunday', win=3600, step=1200)