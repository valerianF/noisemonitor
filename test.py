from modules.utilities import *
from modules.monitor import *

path0 = "D:\\Files\\OneDrive - McGill University\\Work\\Fleurs de Macadam - Musiroy\\Acou_Data\\data_NS\\2020\\May\\Capteur 962_2020_04_28__15h00m00sHHMMSSms.xls"
path1 = "D:\\Files\\OneDrive - McGill University\\Work\\Fleurs de Macadam - Musiroy\\Acou_Data\\data_NS\\2019\\MayJune\\1_962_DBAdata_may_2019.csv"
path2 = "D:\\Files\\OneDrive - McGill University\\Work\\Fleurs de Macadam - Musiroy\\Acou_Data\\data_NS\\2019\\MayJune\\2_962_DBAdata_june_2019.csv"

path3 = "D:\\Files\\OneDrive - McGill University\\Work\\Fleurs de Macadam - Musiroy\Acou_Data\data_NS\\2019\\July\\3_962_DBAdata_july_2019.csv"


path = "D:\\Files\\OneDrive - McGill University\\Work\\Misc\\SoundWalksEdda\\continuous_measurements\\Bruit_et_vibrations-Point_7.xlsx"

df = load_data([path], datetimeindex=1, valueindex=5, header=0)

# time, mean = sliding_mean(df.iloc[:,0], df.iloc[:,2])

Av = LevelMonitor(df)

mean = Av.daily(23, 7, win=3600, step=1200)

mean2 = Av.weekly(0, 23, 'monday', 'friday', win=3600, step=1200)
mean3 = Av.weekly(0, 23, 'saturday', 'sunday', win=3600, step=1200)

mean4 = Av.sliding_average(step=1200)

l1 = Av.lden('monday', 'friday')
l2 = Av.lden('saturday', 'sunday')
l3, d, e, n = Av.lden(values=True)