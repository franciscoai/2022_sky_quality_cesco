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
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.dates as mdates
import matplotlib.colors as colors
from scipy.optimize import curve_fit
from water_vapor import water_vapor
import pysolar.solar as solar


def fit_func(x, a, b, c):
    # to use in fits
    return a*np.exp(x+b)+c


REPO_PATH = os.getcwd()
# 'mica_hourly'  # [str(i)+'0217' for i in range(1997,2013,1)] # Full dates to plot, set to None to plot all
MICAF = 'mica_calibration'  # 'mica_vs_master'# 'mica_hourly' #'mica_vs_master'  # 'mica_hourly'  # ['19990222']
# other options are: 'mica_hourly' to plot the same day in all years
DEL_MICA_MONTHS = ['200507', '200508', '200509', '200510', '200511']  # months to delete
MICA_DIR = REPO_PATH + '/data/mica/Sky_Tes_1997_2012'
MASTER_DIR = REPO_PATH + '/data/master'
OPATH = REPO_PATH + '/output/mica'
COL_NAMES = ['Date', 'Sky_T', 'Sun_T']
COL_UNITS = {'Date': '', 'Sky_T': '', 'Sun_T': '', 'Sky_T_over_Sun_T': '',
             'sky_class': '', 'date_diff': '[s]'}  # units incl a blank space
DEL_VAL = {'Sky_T': [4.91499996], 'Sun_T': [], 'Sky_T_over_Sun_T':[]}  # {'Sky_T': [4.91499996, 0.0], 'Sun_T': [0.0]}  # delete these values
MIN_VAL = {'Sky_T': [], 'Sun_T': [], 'Sky_T_over_Sun_T':[]}  # delet all values below these
NBINS = 50  # his num of bin
CLUSTERS = ['Sunny Good', 'Sunny Moderate', 'Sunny Bad', 'Cloudy']
CLUSTERING_METHOD = 'manual'  # 'kmeans' or 'manual'
N_CLUSTERS = len(CLUSTERS)  # Number of clusters to clasiffy the sky
CLUSTERS_COLOR = ['g', 'orange', 'r', 'k']
EMP_CLASS = {'Sky_T': [1, 2], 'Sun_T': [2, 2]}  # empirical limits for sky classes in mV
DO_NORM = False  # normalize the data
matplotlib.rc('font', size=12)  # font size
BWIDTH = 0.45
DPI = 300.  # image dpi
DATE_DIFF_LIM = [0, 50]  # date difference limit for the plot
RECOMPUTE_SCALER = True  # Set to recompute the data scaler
SCALER_FILE = REPO_PATH + '/output/mica/scaler.pckl'
OAFA_LOC = [-31+48/60.+8.5/3600, -69+19/60.+35.6/3600., 2370.]  # oafa location lat, long, height [m]
MICA_CAL_DIR = '/media/sf_iglesias_data/cesco_sky_quality/MICA_processed/AvgGifs'

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

# Normalizes
if DO_NORM:
    if RECOMPUTE_SCALER and MICAF is None:
        print('Generating and saving normalization parameters for...', COL_NAMES[1:])
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

    lim = np.array([EMP_CLASS['Sky_T'], EMP_CLASS['Sun_T']])
    lim = scaler.transform(lim)
    EMP_CLASS['Sky_T'] = lim[:, 0]
    EMP_CLASS['Sun_T'] = lim[:, 1]
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

# Plots
print('Plotting...')
os.makedirs(OPATH, exist_ok=True)
if MICAF == 'mica_vs_master':
    REMOVE = {"Sky_T": [-100], "Amb_T": [], "WindS": [], "Hum%": [], "DewP": [],
              "C": [], "W": [], "R": [], "Cloud_T": [], "Date_diff": []}  # Max val
    COL_NAMES_MASTER = ["DAY", "HOUR(UT)", "Sky_T", "Amb_T", "WindS", "Hum%",
                        "DewP", "C", "W", "R", "Cloud_T", "Date_diff"]
    mf = [os.path.join(MASTER_DIR, f) for f in os.listdir(MASTER_DIR) if f.endswith('_wea.txt')]
    mf = [i for i in mf if str.split(os.path.basename(i), '.')[0][0:4] in ['2012']]
    df_master = []
    for f in mf:
        df = pd.read_csv(f, delim_whitespace=True, skiprows=1, names=COL_NAMES_MASTER)
        df["DAY"] = pd.to_datetime(df["DAY"], format="%Y%m%d")
        df["HOUR(UT)"] = [timedelta(hours=h) for h in df['HOUR(UT)']]
        df["Date"] = df["DAY"]+df["HOUR(UT)"]
        df_master.append(df)
    df_master = pd.concat(df_master, ignore_index=True)
    df_master["Cloud_T"] = df_master["Sky_T"] - df_master["Amb_T"]
    df_master["water_vapor"] = water_vapor(df_master["Amb_T"], df_master["Hum%"])
    df_master["Date_diff"] = df_master["Date"].diff().dt.total_seconds()
    # remove values
    for var in COL_NAMES_MASTER[2:]:
        for i in REMOVE[var]:
            df_master = df_master.drop(df_master[df_master[var] < i].index)

    # normalizes
    scaler_master = MinMaxScaler()
    tonorm = ['Sky_T', "Amb_T", "WindS", "Hum%", "DewP", "Cloud_T", "water_vapor"]
    df_master[tonorm] = scaler_master.fit_transform(df_master[tonorm])

    for var in tonorm:
        # resamples master to mica times
        # df_all.loc[(df_all['sky_class'] != CLUSTERS[2]) & (df_all['sky_class'] != CLUSTERS[3]), "Date"] #  df_all['Date']#  # sunny good and moderate only
        interp_date = df_all.loc[(df_all['Date'].dt.month == 2), "Date"]  # df_all['Sky_T']# df_all['Date']
        # df_all.loc[(df_all['sky_class'] != CLUSTERS[2]) & (df_all['sky_class'] != CLUSTERS[3]), "Sky_T"] #
        interp_skyt = df_all.loc[(df_all['Date'].dt.month == 2), "Sun_T"]  # df_all['Sky_T'] #df_all['Sky_T']
        interp_cloudt = np.interp(interp_date, df_master["Date"], df_master[var])

        # # scatter
        # x = interp_cloudt
        # y = interp_skyt
        # plt.scatter(x, y, c='b', s=2)
        # plt.xlabel(var)
        # plt.ylabel('Sky_T')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # # 3d scatter
        # y = df_all.loc[(df_all['Date'].dt.month == 2), "Sky_T"]
        # z = interp_cloudt
        # x = df_all.loc[(df_all['Date'].dt.month == 2), "Sun_T"]
        # fig = plt.figure(figsize=[10, 10])
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter3D(x, y, z, c='r', marker='.', s=2)
        # ax.set_xlabel(df_all['Sun_T'].name)
        # ax.set_ylabel(df_all['Sky_T'].name)
        # ax.set_zlabel(var)
        # plt.tight_layout()
        # plt.show()

        # # 2d histogram with fit
        if var in ['water_vapor']:
            # vs Date
            plt.figure(figsize=[10, 6])
            x = interp_date
            y = interp_cloudt
            plt.scatter(x, y, c='g', s=2, label='intp_'+var)
            y = interp_skyt
            plt.scatter(x, y, c='b', s=2, label='Sky_T')
            plt.xlim([min(x), max(x)])
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=[8, 9])
            x = interp_cloudt
            y = interp_skyt
            plt.hist2d(x, y, bins=NBINS, cmap='Greys')  # , norm=matplotlib.colors.LogNorm())
            # # fit
            # if var in ['Cloud_T']:
            #     optp, _ = curve_fit(fit_func, x, y, p0=[-1,60,1])
            #     print(optp)
            #     a, b, c = optp
            #     x_line = np.arange(min(x), max(x), 1)
            #     y_line = fit_func(x_line, a, b, c)
            #     plt.plot(x_line, y_line, '--', color='red')

            plt.xlabel('interp_'+var)
            plt.ylabel('Sky_T')
            plt.title('Correlation coef:' + str(np.corrcoef(x, y)))
            plt.tight_layout()
            plt.show()

else:
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
        if var == 'Sky_T_over_Sun_T':
            ax = plt.gca()
            ax.set_yscale('log')

        plt.xlabel(df_all["Date"].name)
        plt.ylabel(df_all[var].name + COL_UNITS[var])
        plt.tight_layout()
        plt.savefig(OPATH+'/'+var, dpi=DPI)
        plt.close()

        # vs hour of that day
        if MICAF == 'mica_hourly':
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
            plt.close()

            # # vs air mass
            # print(OAFA_LOC)
            # date = datetime(2007, 2, 18, 15, 13, 1, 130320, tzinfo=timezone.utc)
            # altitude_deg = solar.get_altitude(OAFA_LOC[0], OAFA_LOC[1], date)
            # solar.radiation.get_radiation_direct(date, altitude_deg)
            # breakpoint()

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
        plt.close()

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
    plt.close()

    # 2d histogram
    plt.figure(figsize=[8, 9])
    y = df_all['Sky_T']
    x = df_all['Sun_T']
    w = df_all['date_diff']/3600.
    hist, xbins, ybins = np.histogram2d(x, y, bins=NBINS, weights=w)
    plt.imshow(hist.T, extent=[min(xbins), max(xbins), min(ybins), max(ybins)],
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

    # plots bars of sky-class per month
    plt.figure(figsize=[10, 8])
    months = df_all['Date'].dt.month.unique()
    n0 = None
    print('sky_class_per_month:')
    print(months)
    for i in range(len(CLUSTERS)):
        n = []
        for m in months:
            # total time in all CLUSTERs per year
            tot = df_all.loc[df_all['Date'].dt.month == m, 'date_diff'].sum()
            # print(tot)
            if tot == 0:
                tot = 1
            n.append(df_all.loc[(df_all['sky_class'] == CLUSTERS[i]) & (
                df_all['Date'].dt.month == m), 'date_diff'].sum()/tot*100.)
        plt.bar(months, n, color=CLUSTERS_COLOR[i], label=CLUSTERS[i], zorder=3, bottom=n0)
        print(CLUSTERS[i], n)
        if n0 is None:
            n0 = np.array(n)
        else:
            n0 += np.array(n)
    plt.legend()
    plt.grid()
    plt.xlabel('month')
    plt.ylabel('$\%$ of observed time')
    plt.tight_layout()
    plt.savefig(OPATH+'/sky_class_per_month', dpi=DPI)
    plt.close()

    # plots bars of sky-class per year only of sunny time (no clouds)
    print('sky_class_per_year_sunny_time:')
    print(years)
    plt.figure(figsize=[10, 8])
    years = df_all['Date'].dt.year.unique()
    n0 = None
    for i in range(len(CLUSTERS)-1):
        n = []
        for y in years:
            # total number of elements in CLUSTER[0:2] per year
            tot = df_all.loc[(df_all['sky_class'] != CLUSTERS[3]) & (df_all['Date'].dt.year == y), 'date_diff'].sum()
            if tot == 0:
                tot = 1
            n.append(df_all.loc[(df_all['sky_class'] == CLUSTERS[i]) & (
                df_all['Date'].dt.year == y), 'date_diff'].sum()/tot*100.)
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
    plt.close()

    # plots bars of sky-class per month only of sunny time (no clouds)
    print('sky_class_per_month_sunny_time:')
    print(months)
    plt.figure(figsize=[10, 8])
    months = df_all['Date'].dt.month.unique()
    n0 = None
    for i in range(len(CLUSTERS)-1):
        n = []
        for m in months:
            # total number of elements in CLUSTER[0:2] per month
            tot = df_all.loc[(df_all['sky_class'] != CLUSTERS[3]) & (df_all['Date'].dt.month == m), 'date_diff'].sum()
            if tot == 0:
                tot = 1
            n.append(df_all.loc[(df_all['sky_class'] == CLUSTERS[i]) & (
                df_all['Date'].dt.month == m), 'date_diff'].sum()/tot*100.)
        plt.bar(months, n, color=CLUSTERS_COLOR[i], label=CLUSTERS[i], zorder=0, bottom=n0)
        # plt.plot(months,np.zeros(len(n))+np.mean(n),'--',color=CLUSTERS_COLOR[i] ,linewidth=4,zorder=1)
        if n0 is None:
            n0 = np.array(n)
        else:
            n0 += np.array(n)
        print(CLUSTERS[i], n)
    plt.legend()
    plt.grid()
    plt.xlabel('month')
    plt.ylabel('$\%$ of Sunny time')
    plt.tight_layout()
    plt.savefig(OPATH+'/sky_class_per_month_sunny_time', dpi=DPI)
    plt.close()
