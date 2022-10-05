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
