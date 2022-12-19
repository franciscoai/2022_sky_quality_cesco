# -*- coding: utf-8 -*-
import pickle
from lib2to3.pgen2.token import OP
import os
from textwrap import shorten
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
from datetime import timezone
from scipy import signal


def sky_b_func(z, a, b, t):
    # sky brigthness from curved atmosphere model as function of the zenith angle (z[deg]) using Lin & Penn 2004 expression
    R = 6378.1*1000. + 2370.  # Earth's radius plus site altitude [m]
    return b + a*(-R*np.cos(np.deg2rad(z))+np.sqrt(R**2*np.cos(np.deg2rad(z))**2+2*R*t+t**2))


def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.median(y, axis=1)


REPO_PATH = os.getcwd()
# 'mica_hourly'  # [str(i)+'0217' for i in range(1997,2013,1)] # Full dates to plot, set to None to plot all
# 'mica_calibration'  # 'mica_vs_master'# 'mica_hourly' #'mica_vs_master'  # 'mica_hourly'  # ['19990222']
MICAF = None  # 'mica_hourly'  # 'mica_hourly'  # 'mica_calibration'
# other options are: 'mica_hourly' to plot the same day in all years
DEL_MICA_MONTHS = ['200507', '200508', '200509', '200510', '200511']  # months to delete
MICA_DIR = REPO_PATH + '/data/mica/Sky_Tes_1997_2012'
OPATH = REPO_PATH + '/output/mica_clear_time'
COL_NAMES = ['Date', 'Sky_T', 'Sun_T']
COL_UNITS = {'Date': '', 'Sky_T': '', 'Sun_T': '', 'Sky_T_over_Sun_T': '',
             'sky_class': '', 'date_diff': '[s]'}  # units incl a blank space
# {'Sky_T': [4.91499996, 0.0], 'Sun_T': [0.0]}  # delete these values
DEL_VAL = {'Sky_T': [4.91499996], 'Sun_T': [], 'Sky_T_over_Sun_T': []}
MIN_VAL = {'Sky_T': [], 'Sun_T': [], 'Sky_T_over_Sun_T': []}  # delet all values below these
NBINS = 50  # his num of bin
matplotlib.rc('font', size=12)  # font size
DPI = 300.  # image dpi
OAFA_LOC = [-31+48/60.+8.5/3600, -69+19/60.+35.6/3600., 2370.]  # oafa location lat, long, height [m]
MICA_CAL_DIR = '/media/sf_iglesias_data/cesco_sky_quality/MICA_processed/AvgGifs'
CAL_EQ = [1.63, 49.01]  # for Fe XIV C at 6 Sr # [2.83, 47.55]  # for Fe XIV L at 6 Sr
SCATTER_LIGHT = 0.7  # in ppm
NOISE_LIM = 0.04 # 0.019*2  # Change lower limit to consider it produced by a cloud
VAR_TO_USE = 'Sun_T'  # to compute clear time

# get all mica files
mf = [os.path.join(MICA_DIR, f) for f in os.listdir(MICA_DIR) if f.endswith('.txt')]
if MICAF is not None:
    if MICAF == 'mica_hourly':
        allf = [str.split(i, '.')[0] for i in os.listdir(MICA_DIR)]
        allf = [i[4:8] for i in allf]
        # find daates that are in at least 14 years
        allf = [i for i in allf if allf.count(i) > 14]
        mf = [i for i in mf if i.split('/')[-1].split('.')[0][4:8] == allf[3]]
    elif MICAF == 'mica_vs_master':
        mf = [i for i in mf if str.split(os.path.basename(i), '.')[0][0:4] in ['2012']]
    elif MICAF == 'mica_calibration':
        mf_cal = os.listdir(MICA_CAL_DIR)
        mf = [i for i in mf if str.split(os.path.basename(i), '.')[0] in mf_cal]
    else:
        mf = [i for i in mf if str.split(os.path.basename(i), '.')[0] in MICAF]

# read the space separated files with pandas
df_all = []
print('Reading %s files...' % len(mf))
tnf = 0
for f in mf:
    yyyymmdd = f.split('/')[-1].split('.')[0]
    if yyyymmdd[0:6] not in DEL_MICA_MONTHS:
        df = pd.read_csv(f, delim_whitespace=True, skiprows=1, names=COL_NAMES)
        df['Date'] = [datetime(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]), tzinfo=timezone.utc) +
                      timedelta(hours=h) for h in df['Date']]
        df['date_diff'] = df['Date'].diff().dt.total_seconds()
        df_all.append(df)
        tnf += 1
df_all = pd.concat(df_all, ignore_index=True)
df_all = df_all.sort_values(by='Date')
COL_NAMES.append('date_diff')

# Eliminate wrong and sat data
for key in DEL_VAL:
    for val in DEL_VAL[key]:
        print('Deleting...', key, val)
        df_all = df_all.drop(df_all[df_all[key] == val].index)
for key in MIN_VAL:
    for val in MIN_VAL[key]:
        print('Deleting...<=', key, val)
        df_all = df_all.drop(df_all[df_all[key] <= val].index)

# adds column with Sky_T_over_Sun_T
df_all['Sky_T_over_Sun_T'] = df_all['Sky_T']/df_all['Sun_T']
COL_NAMES.append('Sky_T_over_Sun_T')

# computes sky brigthness in ppm
df_all['Imica'] = CAL_EQ[1]*df_all['Sky_T_over_Sun_T']+CAL_EQ[0]-SCATTER_LIGHT
COL_NAMES.append('Imica')

# sort by date
df_all = df_all.sort_values(by='Date')

# prints some info
for var in np.array(COL_NAMES)[[1, 2, 4]]:
    print(var+'-------------:')
    print('Total number of files (days) read: %s' % tnf)
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

# computes clear time fraction as in DKIST
print('Computing Clear Time per day')
i = 0
clear_time = []
date_str = df_all['Date'].dt.strftime('%Y%m%d')
for f in mf:
    yyyymmdd = f.split('/')[-1].split('.')[0]
    df = df_all[date_str == yyyymmdd]
    if len(df) > 10: 
        duration = (df['Date'].iloc[-1] - df['Date'].iloc[0]).total_seconds()/3600.
        if duration > 6:
            #print('for ' + yyyymmdd)
            df = df.resample('5s', label='right', on='Date')[VAR_TO_USE].mean()
            cond_int = []
            all_int = 0
            for tint in range(0, len(df), 60):
                asign = np.array(df[tint:tint+60])
                asign = asign[~np.isnan(asign)]
                if len(asign) > 0:
                    asign = np.abs(np.max(asign) - np.min(asign))
                    if asign > NOISE_LIM:
                        cond_int.append(1)  # cloudy
                    else:
                        cond_int.append(0)  # clear
                all_int += 1  # all intervals
            cdate = datetime(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]), tzinfo=timezone.utc)
            if (len(cond_int) != 0) and ((len(cond_int)/all_int) > 0.9):
                ct = float(len(cond_int)-np.sum(cond_int))/(len(cond_int))
                clear_time.append([cdate, ct])

        # # plt.plot(df_all[date_str == yyyymmdd]['Date'][0:-1], np.diff(df_all[date_str == yyyymmdd][VAR_TO_USE]), '*k') # diff
        # plt.plot(df_all[date_str == yyyymmdd]['Date'], (df_all[date_str == yyyymmdd][VAR_TO_USE]), '*k')
        # # plt.yscale('log')
        # plt.title(yyyymmdd + ' ; Clear Time='+'{:3.5f}'.format(ct))
        # print(ct)
        # #plt.ylim([0, 120])
        # plt.show()

    i += 1
clear_time = np.array(clear_time)
# Plots
print('Plotting...')
os.makedirs(OPATH, exist_ok=True)

fig=plt.figure(figsize=[10, 6])
plt.plot(clear_time[:, 0], clear_time[:, 1], '*k')
plt.xlabel('Date')
plt.ylabel('Clear time')
plt.tight_layout()
plt.grid(True)
ax=plt.gca()
plt.savefig(OPATH+'/ct_vs_date', dpi=DPI)
plt.close()

# clear time hist
fig=plt.figure(figsize=[10, 6])
plt.hist(clear_time[:, 1], bins=100, log=True)
plt.title('Median:'+'{:1.2f}'.format(np.median(clear_time[:, 1])))
plt.xlabel('Clear time')
plt.ylabel('Observec days')
plt.tight_layout()
plt.grid(True)
ax=plt.gca()
plt.savefig(OPATH+'/ct_hist', dpi=DPI)
plt.close()

# clear time vs date
print('Number of days used:'+str(len(clear_time)))
print('dates:', clear_time[0, 0], clear_time[-1, 0])