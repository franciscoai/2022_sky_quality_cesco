# -*- coding: utf-8 -*-
import pickle
from lib2to3.pgen2.token import OP
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta
import copy
from pyparsing import col
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.dates as mdates
import matplotlib.colors as colors

REPO_PATH = os.getcwd()
# 'Fig1'  # [str(i)+'0217' for i in range(1997,2013,1)] # Full dates to plot, set to None to plot all
MICAF = 'Fig2'
# other options are: 'Fig1' to plot the same day in all years
DEL_MICA_MONTHS = ['200507', '200508', '200509', '200510', '200511']  # months to delete
HASTA_DIR = REPO_PATH + '/data/hasta/HASTA_Instrument_FF'
OPATH = REPO_PATH + '/output/hasta'
COL_NAMES = ['Date', 'Flare_Factor', 'Sky_T', 'Sun_T']
COL_UNITS = {'Date': '', 'Flare_Factor': '', 'Sky_T': '', 'Sun_T': '',
             'sky_class': '', 'date_diff': '[s]'}  # units incl a blank space
# {'Sky_T': [4.91499996, 0.0], 'Sun_T': [0.0]}  # delete these values
DEL_VAL = {'Flare_Factor': [], 'Sky_T': [-1], 'Sun_T': [-1]}
MIN_VAL = {'Sky_T': [], 'Sun_T': []}  # delet all values below these
MAX_VAL = {'Flare_Factor': [200], 'Sky_T': [], 'Sun_T': []}  # set all values larger than these to these
NBINS = 50  # his num of bin
CLUSTERS = ['Sunny', 'Cloudy']
CLUSTERING_METHOD = 'manual'  # 'kmeans' or 'manual'
N_CLUSTERS = len(CLUSTERS)  # Number of clusters to clasiffy the sky
CLUSTERS_COLOR = ['g', 'k']
EMP_CLASS = {'Flare_Factor': [200*0.37, 200*0.37], 'Sky_T': [1, 2], 'Sun_T': [2, 2]}  # empirical limits for sky classes in mV
DO_NORM = True  # normalize the data
matplotlib.rc('font', size=12)  # font size
BWIDTH = 0.45
DPI = 300.  # image dpi
DATE_DIFF_LIM = [0, 50]  # date difference limit for the plot
RECOMPUTE_SCALER = True  # Set to recompute the data scaler
SCALER_FILE = REPO_PATH + '/output/hasta/scaler.pckl'

# get all mica files
mf = [os.path.join(HASTA_DIR, f) for f in os.listdir(HASTA_DIR) if f.endswith('.txt')]
if MICAF is not None:
    if MICAF == 'Fig1':
        allf = [str.split(i, '.')[0] for i in os.listdir(HASTA_DIR)]
        allf = [i[4:8] for i in allf]
        # find daates that are in at least 14 years
        allf = [i for i in allf if allf.count(i) > 14]
        mf = [i for i in mf if i.split('/')[-1].split('.')[0][4:8] == allf[3]]
    elif MICAF == 'Fig2':  # get only year 2009
        mf = [i for i in mf if str.split(os.path.basename(i), '.')[0][0:4] in ['2009' ,'2010','2011','2012']]
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

# sets max value
for key in MAX_VAL:
    for val in MAX_VAL[key]:
        print('Setting max value..<=', key, val)
        df_all.loc[df_all[key] >= val,key] = val

# Normalizes
if DO_NORM:
    if RECOMPUTE_SCALER:  # and MICAF != 'Fig2':
        print('Computing and saving normalization parameters for...', COL_NAMES[1:])
        scaler = MinMaxScaler()
        df_all[COL_NAMES[1:]] = scaler.fit_transform(df_all[COL_NAMES[1:]])
        with open(SCALER_FILE, 'wb') as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Reading normalization parameters for...', COL_NAMES[1:])
        with open(SCALER_FILE, 'rb') as handle:
            scaler = pickle.load(handle)
        df_all[COL_NAMES[1:]] = scaler.transform(df_all[COL_NAMES[1:]])
    print('Data Max:', scaler.data_max_, 'Data Min:', scaler.data_min_,  'Scale:',
          scaler.scale_,  'Range:', scaler.data_range_, 'Min:', scaler.min_)

    lim = np.array([EMP_CLASS['Flare_Factor'], EMP_CLASS['Sky_T'], EMP_CLASS['Sun_T']]).transpose()
    lim = scaler.transform(lim)
    EMP_CLASS['Flare_Factor'] = lim[:, 0]    
    EMP_CLASS['Sky_T'] = lim[:, 1]
    EMP_CLASS['Sun_T'] = lim[:, 2]
    print('Normalized empirical limits: ', EMP_CLASS)

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
        if c == CLUSTERS[0]:  # sunny
            df_all.loc[df_all['Flare_Factor'] > EMP_CLASS['Flare_Factor'][0], 'sky_class'] = c
        elif c == CLUSTERS[1]:  # cloudy
            df_all.loc[df_all['Flare_Factor'] <= EMP_CLASS['Flare_Factor'][0], 'sky_class'] = c
COL_NAMES.append('sky_class')
print('Points with sky_class==None:', len(df_all[df_all['sky_class'] == 'None']))

# sort by date
df_all = df_all.sort_values(by='Date')
COL_NAMES.append('date_diff')

# prints some info
for var in np.array(COL_NAMES)[[1, 2, 3, 5]]:
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
    # if var == 'date_diff':
    #    plt.ylim(DATE_DIFF_LIM)
    plt.xlabel(df_all["Date"].name)
    plt.ylabel(df_all[var].name + COL_UNITS[var])
    plt.tight_layout()
    plt.savefig(OPATH+'/'+var, dpi=DPI)
    plt.close()

    # histograms
    plt.figure(figsize=[10, 6])
    if var == 'date_diff':
        plt.hist(y, log=True, bins=np.arange(DATE_DIFF_LIM[1]), color='k', histtype='step')
    else:
        plt.hist(y, log=True, bins=NBINS, color='k', histtype='step')
    if var in EMP_CLASS:
        plt.plot([EMP_CLASS[var][0], EMP_CLASS[var][0]], [0, 1e4], 'k--', linewidth=2)
        plt.plot([EMP_CLASS[var][1], EMP_CLASS[var][1]], [0, 1e4],  'k--', linewidth=2)
    plt.grid()
    plt.ylabel('number of cases')
    plt.xlabel(df_all[var].name + COL_UNITS[var])
    plt.tight_layout()
    plt.savefig(OPATH+'/'+var+'_hist', dpi=DPI)
    plt.close()

    # histograms time weigthed
    plt.figure(figsize=[10, 6])
    if var == 'date_diff':
        plt.hist(y, log=True, bins=np.arange(DATE_DIFF_LIM[1]), color='k', histtype='step')
    else:
        w = df_all['date_diff']/3600.    
        plt.hist(y, log=True, bins=NBINS, color='k', histtype='step', weights=w) 
    if var in EMP_CLASS:
        plt.plot([EMP_CLASS[var][0], EMP_CLASS[var][0]], [0, 1e4], 'k--', linewidth=2)
        plt.plot([EMP_CLASS[var][1], EMP_CLASS[var][1]], [0, 1e4],  'k--', linewidth=2)
    plt.grid()
    plt.ylabel('total time [h]')
    plt.xlabel(df_all[var].name + COL_UNITS[var])
    plt.tight_layout()
    plt.savefig(OPATH+'/'+var+'_hist_time', dpi=DPI)
    plt.close()

"""   # vs hour of that day
    if  MICAF == 'Fig2':
        fig = plt.figure(figsize=[10, 6])
        for f in mf:
            yyyymmdd = f.split('/')[-1].split('.')[0]
            df = df_all[df_all['Date'].dt.strftime('%Y%m%d') == yyyymmdd]
            x = [datetime.combine(datetime.today(), datetime.time(i)) for i in pd.to_datetime(
                df['Date'])]
            y = df[var]
            plt.scatter(x, y, marker='.', s=1, label=yyyymmdd)
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
        plt.close() """

if MICAF == 'Fig2':

    # vs date
    plt.figure(figsize=[10, 10])
    y = df_all['Sun_T']
    x = df_all['Date']
    plt.scatter(x, y, c='r', marker='.', s=2,label='Sun_T')
    y = df_all['Flare_Factor']
    plt.scatter(x, y, c='b', marker='.', s=2, label='Flare_Factor') 
    plt.legend() 
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(OPATH+'/Flare_Factor_and_Sun_T_vs_date', dpi=DPI)
    plt.show()
    plt.close()

    # cross correlation plots
    y = df_all['Sun_T']
    x = df_all['Flare_Factor']
    plt.figure(figsize=[10, 10])
    plt.scatter(x, y, c='r', marker='.', s=2)
    plt.xlabel(df_all['Flare_Factor'].name)
    plt.ylabel(df_all['Sun_T'].name)
    plt.tight_layout()
    plt.savefig(OPATH+'/Flare_Factor_vs_Sun_T', dpi=DPI)
    plt.close()

    y = df_all['Sky_T']
    x = df_all['Flare_Factor']
    plt.figure(figsize=[10, 10])
    plt.scatter(x, y, c='r', marker='.', s=2)
    plt.xlabel(df_all['Flare_Factor'].name)
    plt.ylabel(df_all['Sky_T'].name)
    plt.tight_layout()
    plt.savefig(OPATH+'/Flare_Factor_vs_Sky_T', dpi=DPI)
    plt.close()

    #3d scatter
    y = df_all['Sky_T']
    z = df_all['Flare_Factor']
    x = df_all['Sun_T']
    fig=plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter3D(x, y,z, c='r', marker='.', s=2)
    ax.set_xlabel(df_all['Sun_T'].name)
    ax.set_ylabel(df_all['Sky_T'].name)
    ax.set_zlabel(df_all['Flare_Factor'].name)
    ax.set_xlim([0,1])
    ax.set_ylim([1,0])
    ax.set_zlim([0,1])
    plt.tight_layout()
    plt.savefig(OPATH+'/Flare_Factor_vs_Sky_T_vs_sun_T', dpi=DPI)

    # 2d histogram
    plt.figure(figsize=[8, 9])
    y = df_all['Flare_Factor']
    x = df_all['Sun_T']
    w = df_all['date_diff']/3600.
    hist, xbins, ybins = np.histogram2d(x, y, bins=NBINS, weights=w)
    plt.imshow(hist.T, extent=[min(xbins), max(xbins), min(ybins), max(ybins)],
            origin='lower', norm=colors.LogNorm(vmin=5./60., vmax=30*24), cmap='Greys')
    plt.colorbar(label='Total time [h]')
    plt.xlabel(df_all['Sun_T'].name + COL_UNITS['Sun_T'])
    plt.ylabel(df_all['Flare_Factor'].name + COL_UNITS['Flare_Factor'])
    plt.tight_layout()
    plt.savefig(OPATH+'/Flare_Factor_vs_Sun_T_hist', dpi=DPI)
    plt.close()

    # 2d histogram
    plt.figure(figsize=[8, 9])
    y = df_all['Flare_Factor']
    x = df_all['Sky_T']
    w = df_all['date_diff']/3600.
    hist, xbins, ybins = np.histogram2d(x, y, bins=NBINS, weights=w)
    plt.imshow(hist.T, extent=[min(xbins), max(xbins), min(ybins), max(ybins)],
            origin='lower', norm=colors.LogNorm(vmin=5./60., vmax=30*24), cmap='Greys')
    plt.colorbar(label='Total time [h]')
    plt.xlabel(df_all['Sky_T'].name + COL_UNITS['Sun_T'])
    plt.ylabel(df_all['Flare_Factor'].name + COL_UNITS['Flare_Factor'])
    plt.tight_layout()
    plt.savefig(OPATH+'/Flare_Factor_vs_Sky_T_hist', dpi=DPI)
    plt.close()

# plots bars of sky-class per year
plt.figure(figsize=[10, 8])
years = df_all['Date'].dt.year.unique()
n0 = None
print('sky_class_per_year:')
print(years)
for i in range(len(CLUSTERS)):
    n = []
    for y in years:
        # total time in all CLUSTERs per year
        tot = df_all.loc[df_all['Date'].dt.year == y, 'date_diff'].sum()
        # print(tot)
        if tot == 0:
            tot = 1
        n.append(df_all.loc[(df_all['sky_class'] == CLUSTERS[i]) & (
            df_all['Date'].dt.year == y), 'date_diff'].sum()/tot*100.)
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
plt.close()