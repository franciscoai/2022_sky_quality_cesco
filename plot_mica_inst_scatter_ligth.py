# -*- coding: utf-8 -*-
from genericpath import isfile
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
MICAF = None  # 'mica_calibration'
# other options are: 'mica_hourly' to plot the same day in all years
DEL_MICA_MONTHS = ['200507', '200508', '200509', '200510', '200511']  # months to delete
MICA_DIR = REPO_PATH + '/data/mica/Sky_Tes_1997_2012'
OPATH = REPO_PATH + '/output/mica_inst_scatter_light'
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
CAL_EQ = [2.83, 47.55]  # for Fe XIV at 6 Sr
B_LIM = 50  # Limit of B [ppm] term when fitting the Int vs Zenith Angle function
RES_LIM = 50  # Limit of the fit rms residual [ppm]

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

os.makedirs(OPATH, exist_ok=True)

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
df_all['Sky_B2'] = CAL_EQ[1]*df_all['Sky_T_over_Sun_T']+CAL_EQ[0]
COL_NAMES.append('Sky_B2')

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

var = 'Sky_B2'
date_str = df_all['Date'].dt.strftime('%Y%m%d')

# fits only the morning o all rasonable days
pfile = OPATH+'/'+var+'fit_results.pickle'
if os.path.isfile(pfile):
    print('Readig fits results from ...'+pfile)
    with open(pfile, "rb") as input_file:
        fit_res = pickle.load(input_file)
else:
    lim = 40  # np.percentile(df_all['Sky_B2'], 0.01)
    print('Will fit only days with median Sky_B2<'+str(lim))
    i = 0
    fit_res = []
    inst_scatter = []
    for f in mf:
        print(str(i)+' of '+str(len(mf)))
        yyyymmdd = f.split('/')[-1].split('.')[0]
        df = df_all[date_str == yyyymmdd]
        if df['Sky_B2'].median() < lim:
            x = [90. - solar.get_altitude(OAFA_LOC[0], OAFA_LOC[1],  d.to_pydatetime()) for d in df['Date']]
            y = df[var]
            # keeps only the morning
            y = y[0:np.argmin(x)]
            x = x[0:np.argmin(x)]
            try:
                optp, _ = curve_fit(sky_b_func, x, y, p0=[1., 1., 20])
                a, b, t = optp
                chi2 = np.sqrt(np.mean((sky_b_func(x, a, b, t)-y)**2))
                fit_res.append([np.uint32(yyyymmdd), a, b, t, chi2])
                inst_scatter.append(b)
                x_line = np.arange(min(x), max(x), 1)
                y_line = sky_b_func(x_line, a, b, t)
                print(yyyymmdd, a, b, t, chi2)
            except:
                print('Could not fit ' + yyyymmdd)
        i += 1
    fit_res = np.array(fit_res)
    print('Saving fits results to ' + pfile)
    with open(pfile, 'wb') as handle:
        pickle.dump(fit_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plots
print('Plotting...')
# selects fits
sz_all = np.size(fit_res[:, 0].flatten())
ok_ind = np.argwhere((fit_res[:, 2] < B_LIM) & (fit_res[:, 2] > 0) & (fit_res[:, 4] < RES_LIM)).flatten()

# rms residual hist
plt.figure(figsize=[10, 6])
plt.hist(fit_res[ok_ind, 4], log=False, bins=20, cumulative=True,
         color='k', histtype='step', density=True)  # range(0, 30, 1)
plt.grid()
plt.ylim([0, 1])
plt.ylabel('number of days / '+'{:4}'.format(len(ok_ind)))
plt.xlabel('rms residual [ppm]')
plt.tight_layout()
plt.savefig(OPATH+'/'+var+'_fit_Residual_hist', dpi=DPI)
plt.close()

# B vs date
plt.figure(figsize=[10, 6])
plt.plot(medfilt(fit_res[ok_ind, 2].flatten(), 7))
plt.title('Median/Mean:'+str(np.median(fit_res[ok_ind, 2])) + '/'+str(np.mean(fit_res[ok_ind, 2])))
plt.grid()
plt.yscale('log')
plt.xlabel('Date')
plt.ylabel('$B$ [ppm]')
plt.tight_layout()
plt.savefig(OPATH+'/'+var+'_inst_scatter_all', dpi=DPI)
plt.close()

# B hist
plt.figure(figsize=[10, 6])
plt.hist(fit_res[ok_ind, 2], log=False, bins=NBINS, cumulative=True, color='k', histtype='step', density=True)
plt.title('Median/p90:'+str(np.median(fit_res[ok_ind, 2])) + '/'+str(np.quantile(fit_res[ok_ind, 2], 0.9)))
plt.grid()
plt.ylabel('number of days / '+'{:4}'.format(len(ok_ind)))
plt.xlabel('$B$ [ppm]')
plt.tight_layout()
plt.savefig(OPATH+'/'+var+'_inst_scatter_all_hist', dpi=DPI)
plt.close()

# keeps only the best 1% fits
lim = 1 # np.percentile(fit_res[:, 4], 2)
print('Will plot only days with residual <'+str(lim))
fit_res = fit_res[ok_ind, :]
fit_res = fit_res[np.argwhere(fit_res[:, 4] < lim)[:, 0], :]

# sky_b vs time
i = 0
all_b = []
for f in fit_res[:, 0]:
    strf = str(np.int32(f))
    print(str(i)+' of '+str(np.size(fit_res[:, 0])))
    df = df_all[date_str == strf]
    minang = 90. - solar.get_altitude(OAFA_LOC[0], OAFA_LOC[1],  df['Date'].min().to_pydatetime())
    if len(df) > 7000:  # plot only those with at leat XXh of data
        a, b, t, chi2 = fit_res[i, 1:5]
        all_b.append(b)
        if True:
            fig = plt.figure(figsize=[10, 6])
            x = [90. - solar.get_altitude(OAFA_LOC[0], OAFA_LOC[1],  d.to_pydatetime()) for d in df['Date']]
            y = df[var]  # signal.medfilt(df[var], kernel_size=9)  # FILTRO
            x_line = np.arange(min(x), max(x), 1)
            y_line = sky_b_func(x_line, a, b, t)
            plt.scatter(x, y, marker='.', s=1, label=strf)
            plt.plot(x_line, y_line, '-k')
            print(strf, a, b, t, chi2)
            plt.xlabel('zenith_ang [Deg]')
            plt.ylabel(df_all[var].name + '[ppm]')
            plt.ylim([0, 100])
            plt.title('B = '+str(b)+'[ppm]')
            # plt.xlim([20,22])
            # lgnd = plt.legend(loc='lower right')
            # for handle in lgnd.legendHandles:
            #     handle.set_sizes([40.0])
            plt.tight_layout()
            plt.grid(True, which='both')
            ax = plt.gca()
            plt.savefig(OPATH+'/'+var+'_zenith_'+strf, dpi=DPI)
            plt.close()
    i += 1


# B residual hist
print('Number of used days '+str(len(all_b)))
print('Wich is the best '+str(len(all_b)/sz_all))
plt.figure(figsize=[10, 6])
plt.hist(all_b, log=True, bins=20, cumulative=False, color='k', histtype='step')
plt.title('Median/Std dev.:'+str(np.median(all_b))+'/'+str(np.std(all_b)))
plt.grid()
plt.ylabel('number of days')
plt.xlabel('$B$ [ppm]')
plt.tight_layout()
plt.savefig(OPATH+'/'+var+'_inst_scatter_selected', dpi=DPI)
plt.close()
