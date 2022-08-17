# -*- coding: utf-8 -*-
from cProfile import label
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
import matplotlib.colors as colors

REPO_PATH = os.getcwd()
HASTAF = REPO_PATH + '/data/hasta/HASTA_Stats_2009_2020.txt'
# 'Fig1'  # [str(i)+'0217' for i in range(1997,2013,1)] # Full dates to plot, set to None to plot all
MICAF = 'Fig1'# ['20120625', '20120626', '20120627']
# other options are: 'Fig1' to plot the same day in all years
DEL_MICA_MONTHS = ['200507', '200508', '200509', '200510', '200511']  # months to delete
MICA_DIR = REPO_PATH + '/data/mica/Sky_Tes_1997_2012'
OPATH = REPO_PATH + '/output/mica'
COL_NAMES = ['Date', 'Sky_T', 'Sun_T']
COL_UNITS = {'Date': '', 'Sky_T': ' [mV]', 'Sun_T': ' [mV]',
             'sky_class': '', 'date_diff': '[s]'}  # units incl a blank space
DEL_VAL = {'Sky_T': [], 'Sun_T': []}  # {'Sky_T': [4.91499996, 0.0], 'Sun_T': [0.0]}  # delete these values
MIN_VAL = {'Sky_T': [], 'Sun_T': []}  # delet all values below these
NBINS = 50  # his num of bin
CLUSTERS = ['Sunny Good', 'Sunny Moderate', 'Sunny Bad', 'Cloudy']
CLUSTERING_METHOD = 'manual'  # 'kmeans' or 'manual'
N_CLUSTERS = len(CLUSTERS)  # Number of clusters to clasiffy the sky
CLUSTERS_COLOR = ['g', 'orange', 'r', 'k']
EMP_CLASS = {'Sky_T': [1, 2], 'Sun_T': [2, 2]}  # empirical limits for sky classes in mV
DO_NORM = True  # normalize the data
matplotlib.rc('font', size=12)  # font size
BWIDTH = 0.45
DPI = 300.  # image dpi
DATE_DIFF_LIM = [0, 50]  # date difference limit for the plot

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
tnf = 0
for f in mf:
    yyyymmdd = f.split('/')[-1].split('.')[0]
    if yyyymmdd[0:6] not in DEL_MICA_MONTHS:
        df = pd.read_csv(f, delim_whitespace=True, skiprows=1, names=COL_NAMES)
        df['Date'] = [datetime(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])) +
                      timedelta(hours=h) for h in df['Date']]
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

# Normalizes
if DO_NORM:
    scaler = MinMaxScaler()
    all_scaler = []
    for var in COL_NAMES[1:]:
        print('Normalization parameters for...', var)
        df_all[var] = scaler.fit_transform(df_all[[var]])
        all_scaler.append(scaler)
        print('Max:', scaler.data_max_, 'Min:', scaler.data_min_)
        EMP_CLASS[var] = scaler.transform(np.array(EMP_CLASS[var])[:, None]).flatten()
        print('Normalized empirical limits: ', str(EMP_CLASS[var]))

# Clustering
if CLUSTERING_METHOD == 'kmeans':
    # clustering with KMeans
    km = KMeans(n_clusters=N_CLUSTERS)
    df_all['sky_class'] = km.fit_predict(df_all[['Sun_T', 'Sky_T']])
    nind = np.argsort(km.cluster_centers_[:, 0])
    for i in range(len(nind)):
        df_all['sky_class'] = df_all['sky_class'].replace(int(nind[i]), CLUSTERS[len(CLUSTERS)-1-i])
elif CLUSTERING_METHOD == 'manual':
    km = None
    # manual clustering
    df_all['sky_class'] = copy.deepcopy(df_all['Sun_T'])
    df_all['sky_class'] = 'None'
    print('Points with sky_class==None:', len(df_all[df_all['sky_class'] == 'None']))
    for c in CLUSTERS:
        if c == CLUSTERS[0]:  # good
            df_all.loc[(df_all['Sun_T'] > EMP_CLASS['Sun_T'][0]) & (
                df_all['Sky_T'] <= EMP_CLASS['Sky_T'][0]), 'sky_class'] = c
        elif c == CLUSTERS[1]:  # moderate
            df_all.loc[(df_all['Sun_T'] > EMP_CLASS['Sun_T'][0]) & (df_all['Sky_T'] > EMP_CLASS['Sky_T'][0]) &
                       (df_all['Sky_T'] <= EMP_CLASS['Sky_T'][1]), 'sky_class'] = c
        elif c == CLUSTERS[2]:  # bad
            df_all.loc[(df_all['Sun_T'] > EMP_CLASS['Sun_T'][0]) & (
                df_all['Sky_T'] > EMP_CLASS['Sky_T'][1]), 'sky_class'] = c
        elif c == CLUSTERS[3]:  # cloudy
            df_all.loc[(df_all['Sun_T'] <= EMP_CLASS['Sun_T'][0]), 'sky_class'] = c
COL_NAMES.append('sky_class')
print('Points with sky_class==None:', len(df_all[df_all['sky_class'] == 'None']))

# date difference between consecutive rows in seconds
COL_NAMES.append('date_diff')
df_all['date_diff'] = df_all['Date'].diff().dt.total_seconds()

# prints some info
for var in np.array(COL_NAMES)[[0, 1, 2, 4]]:
    print(var+'-------------:')
    print('Total number of files (days) read: %s' % tnf)
    print('Total number of data points: %s (%s days of net observation)' %
          (len(df_all[var]), len(df_all[var])*5./3600./24.))
    print('Mean: %s' % df_all[var].mean())
    print('Median: %s' % df_all[var].median())
    print('Std: %s' % df_all[var].std())
    print('Min: %s' % df_all[var].min())
    print('Max: %s' % df_all[var].max())

# Plots
print('Plotting...')
os.makedirs(OPATH, exist_ok=True)
for var in COL_NAMES:
    # vs date
    y = df_all[var]
    if var in ['Date', 'date_diff']:
        x = np.arange(len(y))
    else:
        x = df_all["Date"]
    plt.figure(figsize=[10, 6])
    plt.scatter(x, y, c='r', marker='.', s=2)
    if var == 'date_diff':
        plt.ylim(DATE_DIFF_LIM)
    plt.xlabel(df_all["Date"].name)
    plt.ylabel(df_all[var].name + COL_UNITS[var])
    plt.tight_layout()
    plt.savefig(OPATH+'/'+var, dpi=DPI)

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
            if var in EMP_CLASS:
                plt.plot([min(x), max(x)], [EMP_CLASS[var][0], EMP_CLASS[var][0]], 'k-', linewidth=2)
                plt.plot([min(x), max(x)], [EMP_CLASS[var][1], EMP_CLASS[var][1]], 'k-', linewidth=2)
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
        plt.savefig(OPATH+'/'+var+'_hour', dpi=DPI)

    # histograms
    plt.figure(figsize=[10, 6])
    if var == 'date_diff':
        plt.hist(y, log=True, bins=np.arange(DATE_DIFF_LIM[1]), color='k', histtype='step')
    else:
        plt.hist(y, log=True, bins=NBINS, color='k', histtype='step')
    if var in EMP_CLASS:
        plt.plot([EMP_CLASS[var][0], EMP_CLASS[var][0]], [0, 4e6], 'k--', linewidth=2)
        plt.plot([EMP_CLASS[var][1], EMP_CLASS[var][1]], [0, 4e6],  'k--', linewidth=2)
    plt.grid()
    plt.ylabel('number of cases')
    plt.xlabel(df_all[var].name + COL_UNITS[var])
    plt.tight_layout()
    plt.savefig(OPATH+'/'+var+'_hist', dpi=DPI)

# 2d scatter
plt.figure(figsize=[10, 10])
for i in range(N_CLUSTERS):
    y = df_all[df_all['sky_class'] == CLUSTERS[i]]['Sky_T']
    x = df_all[df_all['sky_class'] == CLUSTERS[i]]['Sun_T']
    plt.scatter(x, y, c=CLUSTERS_COLOR[i], marker='.', s=0.1, label=CLUSTERS[i])
    plt.plot([EMP_CLASS['Sun_T'][0], max(x)], [EMP_CLASS['Sky_T'][0], EMP_CLASS['Sky_T'][0]], 'k--', linewidth=2)
    plt.plot([EMP_CLASS['Sun_T'][0], max(x)], [EMP_CLASS['Sky_T'][1], EMP_CLASS['Sky_T'][1]], 'k--', linewidth=2)
    plt.plot([EMP_CLASS['Sun_T'][0], EMP_CLASS['Sun_T'][1]], [min(y), max(y)], 'k--', linewidth=2)
if km is not None:
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='b', marker='*', s=100)
plt.xlabel(df_all['Sun_T'].name + COL_UNITS['Sun_T'])
plt.ylabel(df_all['Sky_T'].name + COL_UNITS['Sky_T'])
plt.tight_layout()
plt.savefig(OPATH+'/Sky_T_vs_Sun_T', dpi=DPI)

# 2d histogram
plt.figure(figsize=[8, 9])
y = df_all['Sky_T']
x = df_all['Sun_T']
hist, xbins, ybins = np.histogram2d(x, y, bins=NBINS)
plt.imshow(hist.T*(5./3600.), extent=[min(xbins), max(xbins), min(ybins), max(ybins)],
           origin='lower', norm=colors.LogNorm(vmin=5./60., vmax=30*24), cmap='Greys')
plt.colorbar(label='Total time [h]')
if km is not None:
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='b', marker='*', s=100)
plt.plot([EMP_CLASS['Sun_T'][0], max(x)], [EMP_CLASS['Sky_T'][0], EMP_CLASS['Sky_T'][0]], 'k--', linewidth=2)
plt.plot([EMP_CLASS['Sun_T'][0], max(x)], [EMP_CLASS['Sky_T'][1], EMP_CLASS['Sky_T'][1]], 'k--', linewidth=2)
plt.plot([EMP_CLASS['Sun_T'][0], EMP_CLASS['Sun_T'][1]], [min(y), max(y)], 'k--', linewidth=2)
plt.xlabel(df_all['Sun_T'].name + COL_UNITS['Sun_T'])
plt.ylabel(df_all['Sky_T'].name + COL_UNITS['Sky_T'])
plt.tight_layout()
plt.savefig(OPATH+'/Sky_T_vs_Sun_T_hist', dpi=DPI)

# plots bars of sky-class per year
plt.figure(figsize=[10, 8])
years = df_all['Date'].dt.year.unique()
n0 = None
print('sky_class_per_year:')
print(years)
for i in range(len(CLUSTERS)):
    n = []
    for y in years:
        # total number of elements in all CLUSTERs per year
        tot = len(df_all[df_all['Date'].dt.year == y])
        if tot == 0:
            tot = 1
        n.append(len(df_all[(df_all['sky_class'] == CLUSTERS[i]) & (
            df_all['Date'].dt.year == y)])*5./3600.)  # /tot*100.)
    plt.bar(years, n, color=CLUSTERS_COLOR[i], label=CLUSTERS[i], zorder=3, bottom=n0)
    print(CLUSTERS[i], n)
    if n0 is None:
        n0 = np.array(n)
    else:
        n0 += np.array(n)
plt.legend()
plt.grid()
plt.xlabel('year')
plt.ylabel('$\%$ of observed time')
plt.tight_layout()
plt.savefig(OPATH+'/sky_class_per_year', dpi=DPI)

# plots bars of sky-class per year only of sunny time (not clouds)
print('sky_class_per_year_sunny_time:')
print(years)
plt.figure(figsize=[10, 8])
years = df_all['Date'].dt.year.unique()
n0 = None
for i in range(len(CLUSTERS)-1):
    n = []
    for y in years:
        # total number of elements in CLUSTER[0:2] per year
        tot = len(df_all[(df_all['sky_class'] != CLUSTERS[3]) & (df_all['Date'].dt.year == y)])
        if tot == 0:
            tot = 1
        n.append(len(df_all[(df_all['sky_class'] == CLUSTERS[i]) & (
            df_all['Date'].dt.year == y)])*5./3600.)  # /tot*100.)
    plt.bar(years, n, color=CLUSTERS_COLOR[i], label=CLUSTERS[i], zorder=0, bottom=n0)
    # plt.plot(years,np.zeros(len(n))+np.mean(n),'--',color=CLUSTERS_COLOR[i] ,linewidth=4,zorder=1)
    if n0 is None:
        n0 = np.array(n)
    else:
        n0 += np.array(n)
    print(CLUSTERS[i], n)
plt.legend()
plt.grid()
plt.xlabel('year')
plt.ylabel('$\%$ of Sunny time')
plt.tight_layout()
plt.savefig(OPATH+'/sky_class_per_year_sunny_time', dpi=DPI)
