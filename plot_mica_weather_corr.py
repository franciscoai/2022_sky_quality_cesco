
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pyparsing import col
from calendar import month
from pickle import TRUE
from re import M
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from water_vapor import water_vapor
import matplotlib.dates as mdates


REPO_PATH = os.getcwd()
# 'mica_hourly'  # [str(i)+'0217' for i in range(1997,2013,1)] # Full dates to plot, set to None to plot all
MICAF = None #'mica_hourly'  # 'mica_outlier'  # 'mica_outlier' 'mica_calibration'  # 'mica_hourly' #'mica_vs_master' # ['19990222']
# other options are: 'mica_hourly' to plot the same day in all years
DEL_MICA_MONTHS = ['200507', '200508', '200509', '200510', '200511']
# ,'201201', '201202', '201204', '201205', '201206']  # months to delete
MICA_DIR = REPO_PATH + '/data/mica/Sky_Tes_1997_2012'
MASTER_DIR = REPO_PATH + '/data/master'
OPATH = REPO_PATH + '/output/mica_calibrated'
COL_NAMES = ['Date', 'Sky-T', 'Sun-T']
COL_UNITS = {'Date': '', 'Sky-T': '[mV]', 'Sun-T': '[mV]', 'Sky-T/Sun-T': '',
             'sky_class': '', 'date_diff': '[s]', 'Imica': '[ppm]'}  # units incl a blank space
# {'Sky-T': [4.91499996, 0.0], 'Sun-T': [0.0]}  # delete these values
DEL_VAL = {'Sky-T': [4.91499996], 'Sun-T': [], 'Sky-T/Sun-T': []}
MIN_VAL = {'Sky-T': [], 'Sun-T': [], 'Sky-T/Sun-T': []}  # delet all values below these
matplotlib.rc('font', size=12)  # font size
BWIDTH = 0.45
DPI = 300.  # image dpi
MICA_CAL_DIR = '/media/sf_iglesias_data/cesco_sky_quality/MICA_processed/AvgGifs'
CAL_EQ = [1.63, 49.01]  # for Fe XIV C at 6 Sr # [2.83, 47.55]  # for Fe XIV L at 6 Sr
SCATTER_LIGHT = 1.0  # in ppm
SUNSPOT_FILE = REPO_PATH + '/data/sunspot_num.pickle'  # to overplot sunspot num
SCIFMT = '{:4.2f}'
OAFA_LOC = [-31+48/60.+8.5/3600, -69+19/60.+35.6/3600., 2370.]  # oafa location lat, long, height [m]
REPO_PATH_MW = os.getcwd()
path = REPO_PATH_MW + '/data/master'
OPATH_W = REPO_PATH_MW + '/output/weather'
path_wea = REPO_PATH_MW + '/data/wea'
OPATH_WEA = REPO_PATH_MW + '/output/wea'
DAY_TIME = [datetime.time(hour=10, minute=0), datetime.time(hour=22, minute=0)]  # daytime interval
COL_NAMES_M= ["DAY", "HOUR(UT)", "Sky_T", "Amb_T", "WindS", "Hum%", "DewP", "C", "W", "R", "Cloud_T", "DATE_DIF", "WV"]
COL_NAMES_WEA = ["DATE", "HOUR(UT)", "TEMP", "PRESS", "HUM", "WSP", "WDIR"]
NBINS = 50
REMOVE = {"Sky_T": [-990], "Amb_T": [], "WindS": [], "Hum%": [], "DewP": [],
          "C": [], "W": [], "R": [], "Cloud_T": [], "DATE_DIF": [], "WV": []}
variables_w = ["TEMP", "PRESS", "HUM", "WSP", "WDIR", "WV", "DATE_DIF"]  # variables to plot
variables=['TEMP', 'PRESS','HUM','WSP','WDIR', 'WV', 'Cloud_T', 'Sky_T',  "DATE_DIF"]
units = ["  [$^\circ C$]", "[mm Hg]","  [%]","  [m$s^{-1}$]","  [Deg]","  [mm]"," "," ","seg"]
remove_weather_min = {"TEMP":[-20], "PRESS": [], "HUM": [0], "WSP": [0], "WDIR": [], "WV": [], "DATE_DIF": [0]}
remove_weather_max = {"TEMP":[40], "PRESS": [], "HUM": [150], "WSP": [40], "WDIR": [], "WV": [], "DATE_DIF": [3600]}

#---------------------------------------------------------Mica--------------------------------------------
# get all mica files
mf = [os.path.join(MICA_DIR, f) for f in os.listdir(MICA_DIR) if f.endswith('.txt')]
if MICAF is not None:
    if MICAF == 'mica_hourly':
        allf = [str.split(i, '.')[0] for i in os.listdir(MICA_DIR)]
        allf = [i[4:8] for i in allf]
        # find daates that are in at least 14 years
        allf = [i for i in allf if allf.count(i) > 14]
        mf = [i for i in mf if i.split('/')[-1].split('.')[0][4:8] == allf[3]]
    elif MICAF == 'mica_calibration':
        mf_cal = os.listdir(MICA_CAL_DIR)
        mf = [i for i in mf if str.split(os.path.basename(i), '.')[0] in mf_cal]
    elif MICAF == 'mica_outlier':  # only 2012
        mf = [i for i in mf if str.split(os.path.basename(i), '.')[0][0:4] in ['2012']]
    else:
        mf = [i for i in mf if str.split(os.path.basename(i), '.')[0] in MICAF]

# read the space separated files with pandas
df_all = []
print('Reading %s files...' % len(mf))
ndays = 0
for f in mf:
    yyyymmdd = f.split('/')[-1].split('.')[0]
    if yyyymmdd[0:6] not in DEL_MICA_MONTHS:
        df = pd.read_csv(f, delim_whitespace=True, skiprows=1, names=COL_NAMES)
        df['Date'] = [datetime(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])) +
                      timedelta(hours=h) for h in df['Date']]
        df['date_diff'] = df['Date'].diff().dt.total_seconds()
        df.loc[df['date_diff'].isna(), 'date_diff'] = 0
        df_all.append(df)
        ndays += 1

df_all = pd.concat(df_all, ignore_index=True)


# adds column with Sky-T/Sun-T
df_all['Sky-T/Sun-T'] = df_all['Sky-T']/df_all['Sun-T']
COL_NAMES.append('Sky-T/Sun-T')

# Eliminate wrong and sat data
for key in DEL_VAL:
    for val in DEL_VAL[key]:
        len1 = len(df_all[key])
        df_all = df_all.drop(df_all[df_all[key] == val].index)
        print('Deleted '+str(len1-len(df_all[key]))+' values of ', key, val)
for key in MIN_VAL:
    for val in MIN_VAL[key]:
        len1 = len(df_all[key])
        df_all = df_all.drop(df_all[df_all[key] <= val].index)
        print('Deleted '+str(len1-len(df_all[key]))+' values of ', key, val)

# computes sky brigthness in ppm
df_all['Imica'] = CAL_EQ[1]*df_all['Sky-T/Sun-T']+CAL_EQ[0]-SCATTER_LIGHT
COL_NAMES.append('Imica')

# sort by date
df_all = df_all.sort_values(by='Date')
COL_NAMES.append('date_diff')

# prints some info
for var in ['Sky-T', 'Sun-T', 'Imica', 'date_diff']:
    print(var+'-------------:')
    print('Total number of files (days) read: %s' % ndays)
    print('Total number of unique datres (days): %s' % np.size(df_all['Date'].dt.date.unique()))
    print('Total number of data points: %s (%s days of net observation)' %
          (len(df_all[var]), len(df_all[var])*5./3600./24.))
    print('Mean: %s' % df_all[var].mean())
    print('Median: %s' % df_all[var].median())
    print('Std: %s' % df_all[var].std())
    print('Min: %s at date %s' % (df_all[var].min(), df_all.loc[df_all[var].idxmin()]['Date']))
    print('Max: %s at date %s' % (df_all[var].max(), df_all.loc[df_all[var].idxmin()]['Date']))
    print('p90: %s' % np.nanpercentile(df_all[var], 90))
    print('p99: %s' % np.nanpercentile(df_all[var], 99))
    print('p10: %s' % np.nanpercentile(df_all[var], 10))
    print('----------------------------------------------------')
#---------------------------------------------Master and Weather----------------------------------------------
# create OPATH_W
os.makedirs(OPATH, exist_ok=True)
mf = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_wea.txt')]

# reading all files and converting to datetime
df_all_master = []
for f in mf:
    df = pd.read_csv(f, delim_whitespace=True, skiprows=1, names=COL_NAMES)
    df["DAY"] = pd.to_datetime(df["DAY"], format="%Y%m%d")
    df["HOUR(UT)"] = [datetime.timedelta(hours=h) for h in df['HOUR(UT)']]
    df["DATE"] = df["DAY"]+df["HOUR(UT)"]
    df_all_master.append(df)
df_all_master = pd.concat(df_all_master, ignore_index=True)

df_all_master["Cloud_T"] = df_all_master["Sky_T"] - df_all_master["Amb_T"]
df_all_master["DATE_DIF"] = df_all_master["DATE"].diff().dt.total_seconds()

# remove values
for var in COL_NAMES[2:]:
    for i in REMOVE[var]:
        df_1 = df_all_master[df_all_master[var] <= i].index
        df_final = df_all_master.drop(df_1)


mf_wea = [os.path.join(path_wea, f) for f in os.listdir(path_wea) if f.endswith('.txt')]
# reading all files and converting to datetime
df_all_wea = []
for g in mf_wea:
    df_wea = pd.read_csv(g, delim_whitespace=True, skiprows=2, names=COL_NAMES_WEA, encoding='latin1')
    df_wea["DATE"] = pd.to_datetime(df_wea["DATE"], format="%Y%m%d")
    df_wea["HOUR(UT)"] = pd.to_timedelta(df_wea["HOUR(UT)"])
    df_wea["DATE_TIME"] = df_wea["DATE"]+df_wea["HOUR(UT)"]
    df_wea['DATE_DIF'] = df_wea['DATE_TIME'].diff().dt.total_seconds()
    df_all_wea.append(df_wea)
df_all_wea = pd.concat(df_all_wea, ignore_index=True)
df_all_wea["WV"] = water_vapor(df_all_wea["Amb_T"], df_all_wea["Hum%"])

# creating df with day hours
df_weather = df_all_wea.loc[(df_all_wea["DATE_TIME"].dt.hour > 9) & (df_all_wea["DATE_TIME"].dt.hour < 22)]

# remove values
for var in variables_w:
    for i in remove_weather_min[var]:
        df_2 = df_weather[df_weather[var] <= i].index
        df_weather = df_weather.drop(df_2)

for var in variables_w:
    for i in remove_weather_max[var]:
        df_3 = df_weather[df_weather[var] >= i].index
        df_weather= df_weather.drop(df_3)

#Renaming and creating columns
#df_final["ORIGIN"] = 'M'
#df_weather["ORIGIN"] = "W"
df_final.rename(columns = {'DATE':'DATE_TIME'}, inplace = True)
df_final.rename(columns = {'DAY':'DATE'}, inplace = True)
df_final.rename(columns = {'Amb_T':'TEMP'}, inplace = True)
df_final.rename(columns = {'Hum%':'HUM'}, inplace = True)
df_final.rename(columns = {'WindS':'WSP'}, inplace = True)
df_final["WV"] = water_vapor(df_final["TEMP"], df_final["HUM"])

#Creating combined dataframe
df_combined = pd.DataFrame
df_combined = pd.concat([df_final,df_weather],ignore_index = True,sort = False)
df_combined.drop(['C'], axis=1, inplace = True)
df_combined.drop(['W'], axis=1, inplace = True)
df_combined.drop(['R'], axis=1, inplace = True)
df_combined.drop(['DewP'], axis=1, inplace = True)
df_combined=df_combined.resample('300S', on='DATE_TIME').mean()
df_combined.reset_index(inplace=True)




############################
#CALCULAR Z:

z = [90. - solar.get_altitude(OAFA_LOC[0], OAFA_LOC[1],  d.to_pydatetime()) for d in df['Date']]

