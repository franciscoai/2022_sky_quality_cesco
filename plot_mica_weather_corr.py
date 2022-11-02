# -*- coding: utf-8 -*-
import pickle
from lib2to3.pgen2.token import OP
import os
from textwrap import shorten
from tkinter.messagebox import NO
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta
import copy
from pyparsing import col
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from water_vapor import water_vapor
import pysolar.solar as solar

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


############################
CALCULAR Z:

z = [90. - solar.get_altitude(OAFA_LOC[0], OAFA_LOC[1],  d.to_pydatetime()) for d in df['Date']]
