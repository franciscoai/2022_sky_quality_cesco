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

REPO_PATH = os.getcwd()
# 'mica_hourly'  # [str(i)+'0217' for i in range(1997,2013,1)] # Full dates to plot, set to None to plot all
# 'mica_hourly'  # 'mica_outlier'  # 'mica_outlier' 'mica_calibration'  # 'mica_hourly' #'mica_vs_master' # ['19990222']
MICAF = None  # 'mica_hourly'
# other options are: 'mica_hourly' to plot the same day in all years
# , '201203', '201202', '201204', '201205', '201206']  # months to delete
# , '201202', '201203', '201204', '201205', '201206']
DEL_MICA_MONTHS = ['199906', '200507', '200508', '200509', '200510', '200511']
MICA_DIR = REPO_PATH + '/data/mica/Sky_Tes_1997_2012'
MASTER_DIR = REPO_PATH + '/data/master'
OPATH = REPO_PATH + '/output/mica_calibrated'
COL_NAMES = ['Date', 'Sky-T', 'Sun-T']
COL_UNITS = {'Date': '', 'Sky-T': '[mV]', 'Sun-T': '[mV]', 'Sky-T/Sun-T': '',
             'sky_class': '', 'date_diff': '[s]', 'Imica': '[ppm]'}  # units incl a blank space
# {'Sky-T': [4.91499996, 0.0], 'Sun-T': [0.0]}  # delete these values
DEL_VAL = {'Sky-T': [4.91499996], 'Sun-T': [], 'Sky-T/Sun-T': []}
MIN_VAL = {'Sky-T': [], 'Sun-T': [], 'Sky-T/Sun-T': []}  # delet all values below these

PLT_LIM = {'Sky-T': [5, 0.5], 'Sun-T': [3.25, 0.25], 'Imica': [120, 5], 'date_diff': [20, 1]}
matplotlib.rc('font', size=12)  # font size
BWIDTH = 0.45
DPI = 300.  # image dpi
MICA_CAL_DIR = '/media/sf_iglesias_data/cesco_sky_quality/MICA_processed/AvgGifs'
CAL_EQ = [1.63, 49.01]  # for Fe XIV C at 6 Sr # [2.83, 47.55]  # for Fe XIV L at 6 Sr
SCATTER_LIGHT = 0.7  # in ppm
SUNSPOT_FILE = REPO_PATH + '/data/sunspot_num.pickle'  # add to plot sunspot num vs Imica
SCIFMT = '{:4.2f}'


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
    elif MICAF == 'mica_outlier2011':  # only 2012
        mf = [i for i in mf if str.split(os.path.basename(i), '.')[0][0:4] in ['2011', '2012']]
    elif MICAF == 'mica_outlier1999':  # only 2012
        mf = [i for i in mf if str.split(os.path.basename(i), '.')[0][0:4] in ['1999']]
    else:
        mf = [i for i in mf if str.split(os.path.basename(i), '.')[0] in MICAF]

# read the space separated files with pandas
df_all = []
print('Reading %s files...' % len(mf))
ndays = 0
for f in mf:
    yyyymmdd = f.split('/')[-1].split('.')[0]
    if yyyymmdd[0:6] not in DEL_MICA_MONTHS:
        print('Reading ' + yyyymmdd)
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

# Plots
print('Plotting...')
os.makedirs(OPATH, exist_ok=True)

# vs hour of that day
if MICAF == 'mica_hourly':
    for var in ['Sky-T', 'Sun-T', 'Imica', 'date_diff']:
        fig = plt.figure(figsize=[10, 6])
        for f in mf:
            yyyymmdd = f.split('/')[-1].split('.')[0]
            df = df_all[df_all['Date'].dt.strftime('%Y%m%d') == yyyymmdd]
            x = [datetime.combine(datetime.today(), datetime.time(i)) for i in pd.to_datetime(
                df['Date'])]
            y = df[var]
            plt.scatter(x, y, marker='.', s=1, label=yyyymmdd)
        if var == 'Imica':
            plt.ylim([0, 120])
        plt.xlabel('Time of day [UTC]')
        plt.ylabel(df_all[var].name + ' ' + COL_UNITS[var])
        plt.yticks(np.arange(0, PLT_LIM[var][0]+PLT_LIM[var][1], PLT_LIM[var][1]))
        plt.tight_layout()
        plt.grid(True)
        ax = plt.gca()
        monthyearFmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(monthyearFmt)
        if var == 'Sun-T':
            #plt.ylim([0.4, 1])
            lgnd = plt.legend(loc='lower right')
            for handle in lgnd.legendHandles:
                handle.set_sizes([40.0])
        plt.savefig(OPATH+'/'+var+'_hour', dpi=DPI)
        plt.close()

# seasonal, yearly and cumulative histograms
months = df_all['Date'].dt.month.unique()
years = np.sort(df_all['Date'].dt.year.unique())

for i in ['Sky-T', 'Sun-T', 'Imica', 'date_diff']:
    print(i)

    # vs date
    y = df_all[i]
    x = df_all["Date"]
    plt.figure(figsize=[10, 6])
    # tdf = df_all.resample('M', on='Date').quantile(0.7)
    # plt.plot(tdf["Date"], tdf[i], '--', color='grey', label='Monthly p70 ('+SCIFMT.format(tdf[i].median())+' ; '+SCIFMT.format(tdf[i].min())+' ; '+SCIFMT.format(tdf[i].max())+')')
    tdf = df_all.resample('M', on='Date').median()
    tdfq = df_all.resample('M', on='Date').quantile(0.1)
    tdf.reset_index(inplace=True)
    plt.plot(tdf["Date"], tdf[i], color='black', label='Monthly median ('+SCIFMT.format(tdf[i].median()) +
             ' ; '+SCIFMT.format(tdf[i].min())+' ; '+SCIFMT.format(tdf[i].max())+')')

    plt.plot(tdf["Date"], tdfq[i], color='grey', label='Monthly p10 ('+SCIFMT.format(tdfq[i].median()) +
             ' ; '+SCIFMT.format(tdfq[i].min())+' ; '+SCIFMT.format(tdfq[i].max())+')')

    # if SUNSPOT_FILE is not None:
    #     with open(SUNSPOT_FILE, 'rb') as handle:
    #         sunspot = pickle.load(handle)
    #     sunspot = sunspot.resample('M', on='date').median()
    #     sunspot.reset_index(inplace=True)
    #     plt.plot(sunspot["date"], 40*sunspot['num']/sunspot['num'].max(),'--k', label='Norm sunspot monthly median')
    plt.xlabel(df_all["Date"].name)
    plt.ylabel(df_all[i].name + COL_UNITS[i])
    plt.tight_layout()
    #plt.ylim([0, 50])
    plt.minorticks_on()
    plt.grid()  # visible=True, which='both')
    plt.legend()
    plt.savefig(OPATH+'/'+i+'_vs_date', dpi=DPI)
    plt.close()

    # vs sunspot num
    if SUNSPOT_FILE is not None and i=='Imica':
        with open(SUNSPOT_FILE, 'rb') as handle:
            sunspot = pickle.load(handle)
        sunspot = sunspot.resample('M', on='date').median()
        sunspot.reset_index(inplace=True)
        common = np.intersect1d(tdf['Date'].dt.to_period('M'), sunspot['date'].dt.to_period('M'))
        sunspot = sunspot.loc[[i in common for i in sunspot['date'].dt.to_period('M')]]  # keeps only common motnhs
        tdf = tdf.loc[[i in common for i in tdf['Date'].dt.to_period('M')]]
        plt.figure(figsize=[10, 6])
        plt.scatter(sunspot['num'], tdf[i])
        plt.xlabel('Sunspot number')
        plt.ylabel(df_all[i].name + COL_UNITS[i])
        plt.tight_layout()
        plt.minorticks_on()
        plt.grid()
        plt.legend()
        plt.savefig(OPATH+'/'+i+'_vs_sunspot', dpi=DPI)
        plt.close()

    # diff, to compute resolution
    plt.figure(figsize=[10, 6])
    ran = [-0.1, 0.1]
    if i == 'Imica':
        ran = [-0.1, 0.1]
    plt.hist(df_all[i].diff(), bins=100, range=ran, log=True)
    resol = df_all[i].diff().abs().sort_values()
    resol = np.min(resol[resol > 0])
    plt.title('Resolution: '+'{:2.4f}'.format(resol) + COL_UNITS[i])
    plt.xlabel('Diff ' + df_all[i].name + COL_UNITS[i])
    plt.ylabel('Freq.')
    plt.savefig(OPATH+'/'+i+'_diff_hist', dpi=DPI)
    plt.close()

    # Multiplot with stats
    # monthly plot
    fig1, (ax1, ax2, ax3) = plt.subplots(3)
    fig1.set_size_inches(10, 10)
    median = []
    colors = '#e4e8f0'
    for m in months:
        tot = df_all.loc[(df_all['Date'].dt.month == m) & (df_all[i] != np.nan), i]
        median.append(tot)
    ax1 = plt.subplot(2, 2, 1)
    bp = ax1.boxplot(median, showfliers=False, patch_artist=True, meanline=True, whis=[5, 95])
    for median in bp['medians']:
        median.set(color='black', linewidth=3)
    for patch in bp['boxes']:
        patch.set_facecolor(color='white')
    ax1.set_ylabel(i+' ' + COL_UNITS[i])
    ax1.set_xticks(months)
    if i == 'Imica':
        ax1.set_ylim([0, 80])
    ax1.set_xticklabels(months)
    ax1.minorticks_on()
    ax1.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax1.yaxis.grid(True, which='major')
    ax1.xaxis.set_tick_params(which='minor', bottom=False)
    ax1.set_xlabel('Month')

    # Ploting Year median and standar deviation
    median_y = []
    median_sunspot = []
    if SUNSPOT_FILE is not None:
        with open(SUNSPOT_FILE, 'rb') as handle:
            sunspot = pickle.load(handle)
    for y in years:
        total = df_all.loc[(df_all['Date'].dt.year == y) & (df_all[i] != np.NAN), i]
        median_y.append(total)
        if SUNSPOT_FILE is not None:
            median_sunspot.append(sunspot.loc[(sunspot['date'].dt.year == y), 'num'].mean())
    ax2 = plt.subplot(2, 2, 2)
    if i == 'Imica':
        for_sctr = np.array([np.median(i) for i in median_y])
    bp2 = ax2.boxplot(median_y, showfliers=False, patch_artist=True, whis=[5, 95])
    for median in bp2['medians']:
        median.set(color='black', linewidth=3)
    for patch in bp2['boxes']:
        patch.set_facecolor(color='white')
    if len(median_sunspot) > 0:
        median_sunspot = np.array(median_sunspot)
        x = np.arange(1, len(years)+1, 1)
        #ax2.plot(x, 40.*median_sunspot/median_sunspot.max(), '--.k')
    ax2.set_ylabel(i+' ' + COL_UNITS[i])
    ax2.set_xlabel('Year')
    if i == 'Imica':
        ax2.set_ylim([0, 80])
    ax2.set_xticklabels([str(y)[2:4] for y in years])
    ax2.minorticks_on()
    ax2.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax2.yaxis.grid(True, which='Major')
    ax2.xaxis.set_tick_params(which='minor', bottom=False)

    # Ploting histogram
    val_medio = round(df_all[i].mean(), 1)
    max = df_all[i].max()
    min = df_all[i].min()
    ax3 = plt.subplot(2, 2, 3)
    if i == 'Imica':
        bins_list = [i for i in range(0, 100, 1)]
        bins_list.append(max)
    else:
        bins_list = 20
    ax3.hist(df_all[i], density=False, weights=df_all["date_diff"]/3600.,
             cumulative=False, color="grey", bins=bins_list, log=False)
    ax3.set_xlabel(i+' ' + COL_UNITS[i])
    ax3.set_ylabel('Observed hours')
    ax3.minorticks_on()
    ax3.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax3.yaxis.grid(True, which='Major')
    ax3.xaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax3.xaxis.grid(True, which='Major')
    #ax3.set_ylim([0, 1200])
    if i == 'Imica':
        ax3.set_xlim([0, 100])
    print("Median: " + SCIFMT.format(df_all[i].median()))
    print("Min: "+SCIFMT.format(min))
    print("p95 ; p5: "+SCIFMT.format(df_all[i].quantile(0.95))+' ; '+SCIFMT.format(df_all[i].quantile(0.05)))

    # Ploting cumulative histogram
    val_medio = round(df_all[i].mean(), 1)
    mediana = df_all[i].median()
    max = df_all[i].max()
    min = df_all[i].min()
    ax3 = plt.subplot(2, 2, 4)
    if i == 'Imica':
        bins_list = [i for i in range(0, 100, 1)]
        bins_list.append(max)
    else:
        bins_list = 20
    ann_hours_coor = 365.*np.size(years)/ndays
    ax3.hist(df_all[i], density=False, weights=df_all["date_diff"]/3600./np.size(years)*ann_hours_coor,
             cumulative=True, color="grey", bins=bins_list, log=False)
    ax3.set_xlabel(i+' ' + COL_UNITS[i])
    ax3.set_ylabel('Anual hours')
    ax3.minorticks_on()
    ax3.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax3.yaxis.grid(True, which='Major')
    ax3.xaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax3.xaxis.grid(True, which='Major')
    #ax3.set_ylim([0, 2000])
    if i == 'Imica':
        ax3.set_xlim([0, 100])
    # ax3.text(0.5,0.5, "Min: "+str(min),horizontalalignment = 'center',verticalalignment='bottom',fontsize = 11)
    # ax3.text("Max: "+str(max),fontsize = 11)
    # ax3.text("Mean value:  "+str(val_medio),fontsize = 11)
    # ax3.text("Median: "+str(mediana),fontsize = 11)

    plt.tight_layout()
    plt.savefig(OPATH+'/'+i, dpi=300)
    plt.close()

fig = plt.figure(figsize=[10, 6])
plt.scatter(median_sunspot, for_sctr)
plt.ylabel('Yearly median Imica [ppm]')
plt.xlabel('Yearly median Sunspot No.')
plt.grid()
#plt.xlim([0, 250])
#plt.ylim([10, 30])
plt.tight_layout()
plt.savefig(OPATH+'/Imica_vs_sunspot_yearly', dpi=300)
plt.close()
