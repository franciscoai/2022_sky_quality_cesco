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
from  scipy import signal

def sky_b_func(z, a, b, t):
    # sky brigthness from curved atmosphere model as function of the zenith angle (z[deg]) using Lin & Penn 2004 expression
    R = 6378.1*1000. + 2370.  # Earth's radius plus site altitude [m]
    return b + a*(-R*np.cos(np.deg2rad(z))+np.sqrt(R**2*np.cos(np.deg2rad(z))**2+2*R*t+t**2))

def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)


REPO_PATH = os.getcwd()
# 'mica_hourly'  # [str(i)+'0217' for i in range(1997,2013,1)] # Full dates to plot, set to None to plot all
# 'mica_calibration'  # 'mica_vs_master'# 'mica_hourly' #'mica_vs_master'  # 'mica_hourly'  # ['19990222']
MICAF = 'mica_hourly'
# other options are: 'mica_hourly' to plot the same day in all years
DEL_MICA_MONTHS = ['200507', '200508', '200509', '200510', '200511']  # months to delete
MICA_DIR = REPO_PATH + '/data/mica/Sky_Tes_1997_2012'
OPATH = REPO_PATH + '/output/mica_calibrated'
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
CAL_EQ = [9.77, 70.35]  # Linear eq to translate skyt/sunt -> ppm at 2 RS

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

# sort by date
df_all = df_all.sort_values(by='Date')
COL_NAMES.append('date_diff')

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

# computes sky brigthness in ppm
df_all['Sky_B2'] = CAL_EQ[1]*df_all['Sky_T_over_Sun_T']+CAL_EQ[0]
COL_NAMES.append('Sky_B2')

# altitude
print('Computing zenith angle...')
df_all['zenith_ang'] = [90. - solar.get_altitude(OAFA_LOC[0], OAFA_LOC[1],  d.to_pydatetime()) for d in df_all['Date']]
COL_NAMES.append('zenith_ang')

# Plots
print('Plotting...')
os.makedirs(OPATH, exist_ok=True)

# sky_b vs time
var = 'Sky_B2'
inst_scatter = []
if MICAF == 'mica_hourly':
    fig = plt.figure(figsize=[10, 6])
    for f in mf:
        yyyymmdd = f.split('/')[-1].split('.')[0]
        df = df_all[df_all['Date'].dt.strftime('%Y%m%d') == yyyymmdd]
        x = df['zenith_ang']
        y = signal.medfilt(df[var], kernel_size=15)
        plt.scatter(x, y, marker='.', s=1, label=yyyymmdd)
        # fits B vs Zenith angle function
        try:
            optp, _ = curve_fit(sky_b_func, x, y, p0=[1., 1., 20])
            a, b, c = optp
            inst_scatter.append(b)
            x_line = np.arange(min(x), max(x), 1)
            y_line = sky_b_func(x_line, a, b, c)
            plt.plot(x_line, y_line, '--', color='red')
            print(a,b,c)
        except:
            print('Could not fit ' + yyyymmdd)

    plt.xlabel('zenith_ang [Deg]')
    plt.ylabel(df_all[var].name + '[ppm]')
    plt.ylim([0, 200])
    plt.tight_layout()
    plt.grid(True)
    ax = plt.gca()
    plt.savefig(OPATH+'/'+var+'_zenith', dpi=DPI)
    plt.close()
