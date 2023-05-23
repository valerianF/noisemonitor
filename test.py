from modules.load import *
from modules.average import *

path = "D:\\Files\\OneDrive - McGill University\\Work\\Fleurs de Macadam - Musiroy\\Acou_Data\\data_NS\\2020\\May\\Capteur 962_2020_04_28__15h00m00sHHMMSSms.xls"

list = load_data([path], header=1, type='NoiseSentry')

time, mean = sliding_mean(list[0].iloc[:,0], list[0].iloc[:,2])