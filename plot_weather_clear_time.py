from calendar import month
from pickle import TRUE
from re import M
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from water_vapor import water_vapor
import matplotlib.dates as mdates
from datetime import timezone

REPO_PATH = os.getcwd()
path = REPO_PATH + '/data/master'
OPATH = REPO_PATH + '/output/master_clear_time'
COL_NAMES = ["DAY", "HOUR(UT)", "Sky_T", "Amb_T", "WindS", "Hum%", "DewP", "C", "W", "R", "Cloud_T", "DATE_DIF", "WV"]
COL_NAMES_WEA = ["DATE", "HOUR(UT)", "TEMP", "PRESS", "HUM", "WSP", "WDIR"]
NBINS = 50
REMOVE = {"Sky_T": [-990], "Amb_T": [], "WindS": [], "Hum%": [], "DewP": [],
          "C": [], "W": [], "R": [], "Cloud_T": [], "DATE_DIF": [], "WV": []}
variables_w = ["TEMP", "PRESS", "HUM", "WSP", "WDIR", "WV", "DATE_DIF"]  # variables to plot
variables = ['TEMP', 'PRESS', 'HUM', 'WSP', 'WDIR', 'WV', 'Cloud_T', 'Sky_T',  "DATE_DIF"]
units = ["  [$^\circ C$]", " [mm Hg]", "  [%]", "  [m$s^{-1}$]", "  [Deg]", "  [mm]", "  [$^\circ C$]", " [mV]", "seg"]
plot_lables = ['Tempreature ', 'Pressure', 'Humidity', 'Wind Speed',
               'Wind Dir.', 'Water Vapor', 'Cloud Temp.', 'Sky_T',  "DATE_DIF"]
remove_weather_min = {"TEMP": [-20], "PRESS": [], "HUM": [0], "WSP": [0], "WDIR": [], "WV": [], "DATE_DIF": [0]}
remove_weather_max = {"TEMP": [40], "PRESS": [], "HUM": [150], "WSP": [40], "WDIR": [], "WV": [], "DATE_DIF": [3600]}
DPI = 300.  # image dpi

# create opath
os.makedirs(OPATH, exist_ok=True)
mf = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_wea.txt')]

# reading all files and converting to datetime
df_all = []
for f in mf:
    df = pd.read_csv(f, delim_whitespace=True, skiprows=1, names=COL_NAMES)
    df["DAY"] = pd.to_datetime(df["DAY"], format="%Y%m%d")
    df["HOUR(UT)"] = [datetime.timedelta(hours=h) for h in df['HOUR(UT)']]
    df["DATE"] = df["DAY"]+df["HOUR(UT)"]
    df_all.append(df)
df_all = pd.concat(df_all, ignore_index=True)

df_all["Cloud_T"] = df_all["Sky_T"] - df_all["Amb_T"]
df_all["DATE_DIF"] = df_all["DATE"].diff().dt.total_seconds()

# remove values
for var in COL_NAMES[2:]:
    for i in REMOVE[var]:
        df_1 = df_all[df_all[var] <= i].index
        df_final = df_all.drop(df_1)

# keeping only day time hours
df_final = df_final.loc[(df_final["DATE"].dt.hour > 11) & (df_final["DATE"].dt.hour < 23)]


# Renaming and creating columns
#df_final["ORIGIN"] = 'M'
#df_weather["ORIGIN"] = "W"
df_final.rename(columns={'DATE': 'DATE_TIME'}, inplace=True)
df_final.rename(columns={'DAY': 'DATE'}, inplace=True)
df_final.rename(columns={'Amb_T': 'TEMP'}, inplace=True)
df_final.rename(columns={'Hum%': 'HUM'}, inplace=True)
df_final.rename(columns={'WindS': 'WSP'}, inplace=True)
df_final["WV"] = water_vapor(df_final["TEMP"], df_final["HUM"])

W = ['DATE_DIF']

for var in W:
    print(var+'------MASTER DFRAME-------------:')
    print('Total number of unique dates (days): %s' % np.size(df_final['DATE_TIME'].dt.date.unique()))
    print('Total number of data points: %s (%s days of net observation)' %
          (len(df_final[var]), len(df_final[var])*5./3600./24.))
    print('Mean: %s' % df_final[var].mean())
    print('Median: %s' % df_final[var].median())
    print('Std: %s' % df_final[var].std())
    print('Min: %s at date %s' % (df_final[var].min(), df_final.loc[df_final[var].idxmin()]['DATE_TIME']))
    print('Max: %s at date %s' % (df_final[var].max(), df_final.loc[df_final[var].idxmin()]['DATE_TIME']))
    print('p90: %s' % np.nanpercentile(df_final[var], 90))
    print('p99: %s' % np.nanpercentile(df_final[var], 99))
    print('p10: %s' % np.nanpercentile(df_final[var], 10))
    print('----------------------------------------------------')
print('Date range:', df_final['DATE_TIME'].min(), df_final['DATE_TIME'].max())


# computes clear time fraction as in DKIST
i = 0
clear_time = []
date_str = df_final['DATE_TIME'].dt.strftime('%Y%m%d')
days = df_final['DATE_TIME'].dt.date.unique()
print('Computing Clear Time fractrion for {} days'.format(len(days)))
for f in days:
    yyyymmdd = f.strftime('%Y%m%d')
    df = df_final[date_str == yyyymmdd]
    if len(df) > 10:
        duration = (df['DATE_TIME'].iloc[-1] - df['DATE_TIME'].iloc[0]).total_seconds()/3600.
        if duration > 6:
            df = df.resample('60s', label='right', on='DATE_TIME')['Cloud_T'].mean()
            cond_int = []
            all_int = 0
            for tint in range(0, len(df), 5):
                asign = np.array(df[tint:tint+5])
                asign = asign[~np.isnan(asign)]
                if len(asign) > 0:
                    mean = np.mean(asign)
                    max_diff = (np.abs(asign-mean) > 1).sum()  # 3*np.std(asign) #
                    if ((mean > -25) or (max_diff > 2)):  # or (mean < -60):  # see merchant_et_al_2008
                        cond_int.append(1)  # cloudy
                    else:
                        cond_int.append(0)  # clear
                        # print(cond_int, len(asign))
                        # plt.plot(asign)
                        # plt.show()
                all_int += 1  # all intervals
            cdate = datetime.datetime(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]), tzinfo=timezone.utc)
            if (len(cond_int) != 0) and ((len(cond_int)/all_int) > 0.9):
                ct = float(len(cond_int)-np.sum(cond_int))/(len(cond_int))
                clear_time.append([cdate, ct])
                print('day ' + yyyymmdd + ';  ctf: ' + str(ct))
                # if ct < 1:
                #     plt.plot(df)
                #     plt.title('Clear time fraction = ' + str(ct))
                #     plt.show()

    i += 1
clear_time = np.array(clear_time)
# Plots
print('Plotting...')
os.makedirs(OPATH, exist_ok=True)

fig = plt.figure(figsize=[10, 6])
plt.plot(clear_time[:, 0], clear_time[:, 1], '.', markersize=5, color='k')
plt.xlabel('Date')
plt.ylabel('Clear time fraction')
plt.tight_layout()
plt.grid(True)
ax = plt.gca()
plt.savefig(OPATH+'/ct_vs_date', dpi=DPI)
plt.close()

# clear time hist
fig = plt.figure(figsize=[10, 6])
plt.hist(clear_time[:, 1], bins=100, log=True)
plt.title('Median:'+'{:1.2f}'.format(np.median(clear_time[:, 1])) +
          ' ; Mean:'+'{:1.2f}'.format(np.mean(clear_time[:, 1])))
plt.xlabel('Clear time fraction')
plt.ylabel('Observec days')
plt.tight_layout()
plt.grid(True)
ax = plt.gca()
plt.savefig(OPATH+'/ct_hist', dpi=DPI)
plt.close()

# clear time vs date
print('Number of days used:'+str(len(clear_time)))
print('dates:', clear_time[0, 0], clear_time[-1, 0])

months = df_final['DATE_TIME'].dt.month.unique()
years = df_final['DATE_TIME'].dt.year.unique()

fig1, (ax1, ax2, ax3) = plt.subplots(3)
fig1.set_size_inches(9, 4)
median = []
colors = '#e4e8f0'
for m in months:
    tot = np.argwhere([i.month for i in clear_time[:, 0]] == m)
    median.append(clear_time[tot, 1].flatten())
ax1 = plt.subplot(1, 2, 1)
bp = ax1.boxplot(median, showfliers=False, patch_artist=True)
for median in bp['medians']:
    median.set(color='black', linewidth=3)
for patch in bp['boxes']:
    patch.set_facecolor(color='white')
ax1.set_ylabel('Clear time fraction')
ax1.set_xticks(months)
ax1.set_xticklabels(months)
ax1.minorticks_on()
ax1.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
ax1.yaxis.grid(True, which='major')
ax1.xaxis.set_tick_params(which='minor', bottom=False)
ax1.set_xlabel('Month')

# # Ploting Year median and standar deviation
# median_y = []
# for y in years:
#     tot = np.argwhere([i.year for i in clear_time[:, 0]] == y)
#     median_y.append(clear_time[tot, 1].flatten())
# ax2 = plt.subplot(1, 3, 2)
# nyear = np.arange(len(years))
# bp2 = ax2.boxplot(median_y, showfliers=False, patch_artist=True)
# for median in bp2['medians']:
#     median.set(color='black', linewidth=3)
# for patch in bp2['boxes']:
#     patch.set_facecolor(color='white')
# ax2.set_ylabel('Clear time fraction')
# ax2.set_xticklabels([str(y)[2:4] for y in years])
# ax2.minorticks_on()
# ax2.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
# ax2.yaxis.grid(True, which='Major')
# ax2.xaxis.set_tick_params(which='minor', bottom=False)
# plt.xticks(rotation=45)
# ax2.set_xlabel('Year')


# Ploting cumulative histogram
ax3 = plt.subplot(1, 2, 2)
ax3.hist(clear_time[:, 1], density=True, cumulative=True, color="grey", bins=NBINS, log=False)
ax3.set_xlabel('Clear time fraction')
ax3.minorticks_on()
ax3.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
ax3.yaxis.grid(True, which='Major')
ax3.xaxis.grid(which="minor", linestyle=':', linewidth=0.7)
ax3.xaxis.grid(True, which='Major')
plt.xlim([0,1])
plt.tight_layout()
plt.savefig(OPATH+'/clear_time_fraction', dpi=300)
plt.close()
