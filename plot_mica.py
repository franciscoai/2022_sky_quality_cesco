# -*- coding: utf-8 -*-
from calendar import month
from curses import color_content
from lib2to3.pgen2.token import OP
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta
import copy
import matplotlib.colors as mcolors
from pyparsing import col
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.dates as mdates

REPO_PATH = os.getcwd()
HASTAF = REPO_PATH + '/data/hasta/HASTA_Stats_2009_2020.txt'
MICAF = None  # 'Fig1'  # [str(i)+'0217' for i in range(1997,2013,1)] # Full dates to plot, set to None to plot all
# other options are: 'Fig1' to plot the same day in all years
DEL_MICAF = ['200507', '200508', '200509', '200510', '200511']  # months to delete
MICA_DIR = REPO_PATH + '/data/mica/Sky_Tes_1997_2012'
OPATH = REPO_PATH + '/output/mica'
COL_NAMES = ['Date', 'Sky_T', 'Sun_T']
COL_UNITS = {'Sky_T': ' [mV]', 'Sun_T': ' [mV]'}  # units incl a blank space
DEL_VAL = {'Sky_T': [4.91499996, 0.0], 'Sun_T': [0.0]}  # delete these values
MIN_VAL = {'Sky_T': [], 'Sun_T': [0.018]}  # delet all values below these
NBINS = 50  # his num of bin
CLUSTERS = ['Cloudy', 'High scattering', 'Moderate scattering', 'Low scattering']
N_CLUSTERS = len(CLUSTERS)  # Number of clusters to clasiffy the sky
CLUSTERS_COLOR = ['k', 'r', 'orange', 'g']
EMP_CLASS = {'Sky_T': [1, 2], 'Sun_T': [2, 2]}  # empirical limits for sky classes
DO_NORM = False  # normalize the data
matplotlib.rc('font', size=12)

# get all mica files
mf = [os.path.join(MICA_DIR, f) for f in os.listdir(MICA_DIR) if f.endswith('.txt')]
if MICAF is not None:
    if MICAF == 'Fig1':
        allf = [str.split(i, '.')[0] for i in os.listdir(MICA_DIR)]
        allf = [i[4:8] for i in allf]
        # find daates that are in at least 14 years
        allf = [i for i in allf if allf.count(i) > 14]
        mf = [i for i in mf if i.split('/')[-1].split('.')[0][4:8] == allf[3]]
    else:
        mf = [i for i in mf if str.split(os.path.basename(i), '.')[0] in MICAF]

# read the space separated files with pandas
df_all = []
print('Reading %s files...' % len(mf))
for f in mf:
    yyyymmdd = f.split('/')[-1].split('.')[0]
    if yyyymmdd[0:6] not in DEL_MICAF:
        df = pd.read_csv(f, delim_whitespace=True, skiprows=1, names=COL_NAMES)
        df['Date'] = [datetime(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])) +
                      timedelta(hours=h) for h in df['Date']]
        df_all.append(df)

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

# prints some info
for var in COL_NAMES:
    print(var+':')
    print('Total number of data points: %s' % (len(df_all[var])))
    print('min, max:', df_all[var].min(), df_all[var].max())
    print('mean, std:', df_all[var].mean(), df_all[var].std())
    print('median:', df_all[var].median())

# Normalizes
if DO_NORM:
    scaler = MinMaxScaler()
    all_scaler = []
    for var in COL_NAMES[1:]:
        print('Normalization parameters for...', var)
        df_all[var] = scaler.fit_transform(df_all[[var]])
        all_scaler.append(scaler)
        print('Max:', scaler.data_max_, 'Min:', scaler.data_min_)
        print('Normalized empirical limits: ', str(scaler.transform(np.array(EMP_CLASS[var])[:, None])))

# clustering with KMeans
km = KMeans(n_clusters=N_CLUSTERS)
df_all['sky_class'] = km.fit_predict(df_all[['Sun_T', 'Sky_T']])
nind = np.argsort(km.cluster_centers_[:, 0])
for i in range(len(nind)):
    df_all['sky_class'] = df_all['sky_class'].replace(int(nind[i]), CLUSTERS[i])

# Plots
print('Plotting...')
os.makedirs(OPATH, exist_ok=True)
for var in COL_NAMES[1:]:
    # vs date
    x = df_all["Date"]
    y = df_all[var]
    plt.figure(figsize=[10, 6])
    plt.scatter(x, y, c='r', marker='.', s=2)
    plt.xlabel(df_all["Date"].name)
    plt.ylabel(df_all[var].name + COL_UNITS[var])
    plt.tight_layout()
    plt.savefig(OPATH+'/'+var, dpi=300)

    if len(mf) < 20:
        # vs hour of that day
        fig = plt.figure(figsize=[10, 6])
        for f in mf:
            yyyymmdd = f.split('/')[-1].split('.')[0]
            df = df_all[df_all['Date'].dt.strftime('%Y%m%d') == yyyymmdd]
            x = [datetime.combine(datetime.today(), datetime.time(i)) for i in pd.to_datetime(
                df['Date'])]
            y = df[var]
            plt.scatter(x, y, marker='.', s=1, label=yyyymmdd)
            plt.plot([min(x), max(x)], [EMP_CLASS[var][0], EMP_CLASS[var][0]], 'k--', linewidth=2)
            plt.plot([min(x), max(x)], [EMP_CLASS[var][1], EMP_CLASS[var][1]], 'k--', linewidth=2)
        plt.xlabel('Hour of the day')
        plt.ylabel(df_all[var].name + COL_UNITS[var])
        plt.tight_layout()
        plt.grid(True)
        ax = plt.gca()
        monthyearFmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(monthyearFmt)
        if var == 'Sun_T':
            #plt.ylim([0.4, 1])
            lgnd = plt.legend(loc='lower right')
            for handle in lgnd.legendHandles:
                handle.set_sizes([40.0])
        plt.savefig(OPATH+'/'+var+'_hour', dpi=300)

    # histograms
    plt.figure(figsize=[10, 6])
    plt.hist(y, log=True, bins=NBINS, color='k', histtype='step')
    plt.plot([EMP_CLASS[var][0], EMP_CLASS[var][0]], [0, 1e3], 'k--', linewidth=2)
    plt.plot([EMP_CLASS[var][1], EMP_CLASS[var][1]], [0, 1e3],  'k--', linewidth=2)
    plt.grid()
    plt.ylabel('number of cases')
    plt.xlabel(df_all[var].name + COL_UNITS[var])
    plt.tight_layout()
    plt.savefig(OPATH+'/'+var+'_hist', dpi=300)

# scatter
plt.figure(figsize=[10, 10])
for i in range(N_CLUSTERS):
    y = df_all[df_all['sky_class'] == CLUSTERS[i]]['Sky_T']
    x = df_all[df_all['sky_class'] == CLUSTERS[i]]['Sun_T']
    plt.scatter(x, y, c=CLUSTERS_COLOR[i], marker='.', s=0.1, label=CLUSTERS[i])
    plt.plot([min(x), max(x)], [EMP_CLASS['Sky_T'][0], EMP_CLASS['Sky_T'][0]], 'k--', linewidth=2)
    plt.plot([min(x), max(x)], [EMP_CLASS['Sky_T'][1], EMP_CLASS['Sky_T'][1]], 'k--', linewidth=2)
    plt.plot([EMP_CLASS['Sun_T'][0], EMP_CLASS['Sun_T'][1]], [min(y), max(y)], 'k--', linewidth=2)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='b', marker='*', s=100)
plt.xlabel(df_all['Sun_T'].name + COL_UNITS['Sun_T'])
plt.ylabel(df_all['Sky_T'].name + COL_UNITS['Sky_T'])
plt.tight_layout()
plt.savefig(OPATH+'/Sky_T_vs_Sun_T', dpi=300)

# 2d histogram
plt.figure(figsize=[10, 8])
y = df_all['Sky_T']
x = df_all['Sun_T']
plt.hist2d(x, y, bins=NBINS, cmin=3600./5./6, cmax=7.*24.*3600./5., norm=mcolors.LogNorm(), cmap='Greys')
plt.colorbar()
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='b', marker='*', s=100)
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.xlabel(df_all['Sun_T'].name + COL_UNITS['Sun_T'])
plt.ylabel(df_all['Sky_T'].name + COL_UNITS['Sky_T'])
plt.tight_layout()
plt.savefig(OPATH+'/Sky_T_vs_Sun_T_hist', dpi=300)
