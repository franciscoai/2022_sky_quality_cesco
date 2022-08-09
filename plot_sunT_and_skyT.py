# -*- coding: utf-8 -*-
from curses import color_content
from lib2to3.pgen2.token import OP
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta
import copy
import matplotlib.colors as mcolors
from pyparsing import col
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

REPO_PATH = os.getcwd()
HASTAF = REPO_PATH + '/data/hasta/HASTA_Stats_2009_2020.txt'
MICAF = [10, 11, 12, 13]  # np.arange(365*3) + 365*4  # list with mica files to read. Set to None to read all
DEL_MICAF = ['200507', '200508', '200509', '200510', '200511']  # months to delete
MICA_DIR = REPO_PATH + '/data/mica/Sky_Tes_1997_2012'
OPATH = REPO_PATH + '/output/mica'
COL_NAMES = ['Date', 'Sky_T', 'Sun_T']
COL_UNITS = {'Sky_T': ' [mV]', 'Sun_T': ' [mV]'}  # units incl a blank space
DEL_VAL = {'Sky_T': [4.91499996, 0.0], 'Sun_T': [0.0]}  # delete these values
MIN_VAL = {'Sky_T': [], 'Sun_T': [0.015]}  # delet all values below these
NBINS = 50  # his num of bin
CLUSTERS = ['Cloudy', 'Bad', 'Moderate', 'Good']
N_CLUSTERS = len(CLUSTERS)  # Number of clusters to clasiffy the sky
CLUSTERS_COLOR = ['k', 'r', 'orange', 'g']

# get all mica files
mf = [os.path.join(MICA_DIR, f) for f in os.listdir(MICA_DIR) if f.endswith('.txt')]
if MICAF is not None:
    mf = [mf[i] for i in MICAF]

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

# Normalizes
scaler = MinMaxScaler()
for var in COL_NAMES[1:]:
    print('Normalazing...', var)
    df_all[var] = scaler.fit_transform(df_all[[var]])

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
    plt.savefig(OPATH+'/'+var)

    # histograms
    plt.figure(figsize=[10, 8])
    plt.hist(y, log=True, bins=NBINS, color='k', histtype='step')
    plt.grid()
    plt.ylabel('number of cases')
    plt.xlabel(df_all[var].name + COL_UNITS[var])
    plt.tight_layout()
    plt.savefig(OPATH+'/'+var+'_hist')

# scatter
print(df_all['sky_class'] )
plt.figure(figsize=[10, 10])
for i in range(N_CLUSTERS):
    y = df_all[df_all['sky_class'] == CLUSTERS[i]]['Sky_T']
    x = df_all[df_all['sky_class'] == CLUSTERS[i]]['Sun_T']
    plt.scatter(x, y, c=CLUSTERS_COLOR[i], marker='.', s=0.1, label=CLUSTERS[i])
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='b', marker='*', s=100)
plt.xlabel(df_all['Sun_T'].name + COL_UNITS['Sun_T'])
plt.ylabel(df_all['Sky_T'].name + COL_UNITS['Sky_T'])
plt.tight_layout()
plt.savefig(OPATH+'/Sky_T_vs_Sun_T')

# 2d histogram
plt.figure(figsize=[10, 8])
y = df_all['Sky_T']
x = df_all['Sun_T']
plt.hist2d(x, y, bins=NBINS, cmin=3600./5./6, cmax=7.*24.*3600./5., norm=mcolors.LogNorm(), cmap='Greys')
plt.colorbar()
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='b', marker='*', s=100)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(df_all['Sun_T'].name + COL_UNITS['Sun_T'])
plt.ylabel(df_all['Sky_T'].name + COL_UNITS['Sky_T'])
plt.tight_layout()
plt.savefig(OPATH+'/Sky_T_vs_Sun_T_hist')
