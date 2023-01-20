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


REPO_PATH = os.getcwd()
path = REPO_PATH + '/data/master'
OPATH = REPO_PATH + '/output/weather'
path_wea = REPO_PATH + '/data/wea'
OPATH_WEA = REPO_PATH + '/output/wea'
DAY_TIME = [datetime.time(hour=10, minute=0), datetime.time(hour=22, minute=0)]  # daytime interval
COL_NAMES = ["DAY", "HOUR(UT)", "Sky_T", "Amb_T", "WindS", "Hum%", "DewP", "C", "W", "R", "Cloud_T", "DATE_DIF", "WV"]
COL_NAMES_WEA = ["DATE", "HOUR(UT)", "TEMP", "PRESS", "HUM", "WSP", "WDIR"]
NBINS = 50
REMOVE = {"Sky_T": [-990], "Amb_T": [], "WindS": [], "Hum%": [], "DewP": [],
          "C": [], "W": [], "R": [], "Cloud_T": [], "DATE_DIF": [], "WV": []}  # in
variables_w = ["TEMP", "PRESS", "HUM", "WSP", "WDIR", "WV", "DATE_DIF"]  # variables to plot
variables = ['TEMP', 'PRESS', 'HUM', 'WSP', 'WDIR', 'WV', 'Cloud_T', 'Sky_T',  "DATE_DIF"]
units = ["  [$^\circ C$]", " [mm Hg]", "  [%]", "  [m$s^{-1}$]", "  [Deg]", "  [mm]", "  [$^\circ C$]", " [mV]", "seg"]
plot_lables = ['Tempreature ', 'Pressure', 'Humidity', 'Wind Speed',
               'Wind Dir.', 'Water Vapor', 'Cloud Temp.', 'Sky_T',  "DATE_DIF"]
# in both datasets
remove_weather_min = {"TEMP": [-30], "PRESS": [], "HUM": [0], "WSP": [0], "WDIR": [], "WV": [], "DATE_DIF": [0]}
remove_weather_max = {"TEMP": [50], "PRESS": [], "HUM": [200], "WSP": [50], "WDIR": [], "WV": [], "DATE_DIF": [3600]}


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

df_final.rename(columns={'DATE': 'DATE_TIME'}, inplace=True)
df_final.rename(columns={'DAY': 'DATE'}, inplace=True)
df_final.rename(columns={'Amb_T': 'TEMP'}, inplace=True)
df_final.rename(columns={'Hum%': 'HUM'}, inplace=True)
df_final.rename(columns={'WindS': 'WSP'}, inplace=True)

# remove values
for var in variables[0:4]:
    for i in remove_weather_min[var]:
        df_2 = df_final[df_final[var] <= i].index
        df_final = df_final.drop(df_2)

for var in variables[0:4]:
    for i in remove_weather_max[var]:
        df_3 = df_final[df_final[var] >= i].index
        df_final = df_final.drop(df_3)

# reading weather data
mf_wea = [os.path.join(path_wea, f) for f in os.listdir(path_wea) if f.endswith('.txt')]

# reading all files and converting to datetime
df_all_wea = []
for g in mf_wea:
    df_wea = pd.read_csv(g, delim_whitespace=True, skiprows=2, names=COL_NAMES_WEA, encoding='latin1')
    df_wea["DATE"] = pd.to_datetime(df_wea["DATE"], format="%Y%m%d")
    df_wea["HOUR(UT)"] = pd.to_timedelta(df_wea["HOUR(UT)"])
    df_wea["DATE_TIME"] = df_wea["DATE"]+df_wea["HOUR(UT)"]
    df_wea['DATE_DIF'] = df_wea['DATE_TIME'].diff().dt.total_seconds()
    df_all_wea.append(df_wea)
df_all_wea = pd.concat(df_all_wea, ignore_index=True)
df_all_wea["WV"] = water_vapor(df_all_wea["TEMP"], df_all_wea["HUM"])


# keeping only day time hours
df_weather = df_all_wea.loc[(df_all_wea["DATE_TIME"].dt.hour > 11) & (df_all_wea["DATE_TIME"].dt.hour < 23)]
df_final = df_final.loc[(df_final["DATE_TIME"].dt.hour > 11) & (df_final["DATE_TIME"].dt.hour < 23)]

# remove values
for var in variables_w:
    for i in remove_weather_min[var]:
        df_2 = df_weather[df_weather[var] <= i].index
        df_weather = df_weather.drop(df_2)

for var in variables_w:
    for i in remove_weather_max[var]:
        df_3 = df_weather[df_weather[var] >= i].index
        df_weather = df_weather.drop(df_3)

# Renaming and creating columns
df_final["WV"] = water_vapor(df_final["TEMP"], df_final["HUM"])

W = ['DATE_DIF']
for var in W:
    print(var+'--------WEATHER DFRAME-------------:')
    print('Total number of unique dates (days): %s' % np.size(df_weather['DATE_TIME'].dt.date.unique()))
    print('Total number of data points: %s (%s days of net observation)' %
          (len(df_weather[var]), len(df_weather[var])*5./3600./24.))
    print('Mean: %s' % df_weather[var].mean())
    print('Median: %s' % df_weather[var].median())
    print('Std: %s' % df_weather[var].std())
    print('Min: %s at date %s' % (df_weather[var].min(), df_weather.loc[df_weather[var].idxmin()]['DATE_TIME']))
    print('Max: %s at date %s' % (df_weather[var].max(), df_weather.loc[df_weather[var].idxmin()]['DATE_TIME']))
    print('p90: %s' % np.nanpercentile(df_weather[var], 90))
    print('p99: %s' % np.nanpercentile(df_weather[var], 99))
    print('p10: %s' % np.nanpercentile(df_weather[var], 10))

print('Date range:', df_weather['DATE_TIME'].min(), df_weather['DATE_TIME'].max())

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

# Creating combined dataframe
df_combined = pd.DataFrame
df_combined = pd.concat([df_final, df_weather], ignore_index=True, sort=False)
df_combined.drop(['C'], axis=1, inplace=True)
df_combined.drop(['W'], axis=1, inplace=True)
df_combined.drop(['R'], axis=1, inplace=True)
df_combined.drop(['DewP'], axis=1, inplace=True)
df_combined = df_combined.resample('300S', on='DATE_TIME').mean()
df_combined.reset_index(inplace=True)

W = ['TEMP', 'PRESS', 'HUM', 'WSP', 'WDIR', 'WV', 'DATE_DIF']
for var in W:
    print(var+'--------COMBINED DFRAME-------------:')
    print('Total number of unique dates (days): %s' % np.size(df_weather['DATE_TIME'].dt.date.unique()))
    print('Total number of data points: %s (%s days of net observation)' %
          (len(df_weather[var]), len(df_weather[var])*5./3600./24.))
    print('Mean: %s' % df_weather[var].mean())
    print('Median: %s' % df_weather[var].median())
    print('Std: %s' % df_weather[var].std())
    print('Min: %s at date %s' % (df_weather[var].min(), df_weather.loc[df_weather[var].idxmin()]['DATE_TIME']))
    print('Max: %s at date %s' % (df_weather[var].max(), df_weather.loc[df_weather[var].idxmin()]['DATE_TIME']))
    print('p90: %s' % np.nanpercentile(df_weather[var], 90))
    print('p99: %s' % np.nanpercentile(df_weather[var], 99))
    print('p10: %s' % np.nanpercentile(df_weather[var], 10))
print('Date range:', df_combined['DATE_TIME'].min(), df_combined['DATE_TIME'].max())

months = df_combined['DATE_TIME'].dt.month.unique()
years = np.sort(df_combined['DATE_TIME'].dt.year.unique())

j = 0
# Ploting graphics
for i in variables:
    print(i)
    fig1, (ax1, ax2, ax3) = plt.subplots(3)
    fig1.set_size_inches(13, 4)
    median = []
    colors = '#e4e8f0'
    for m in months:
        tot = df_combined.loc[(df_combined['DATE_TIME'].dt.month == m) & (pd.notna(df_combined[i])), i]
        median.append(tot)

    ax1 = plt.subplot(1, 3, 1)
    bp = ax1.boxplot(median, showfliers=False, patch_artist=True)
    for median in bp['medians']:
        median.set(color='black', linewidth=3)
    for patch in bp['boxes']:
        patch.set_facecolor(color='white')
    ax1.set_ylabel(plot_lables[j]+units[j])
    ax1.set_xticks(months)
    ax1.set_xticklabels(months)
    ax1.minorticks_on()
    ax1.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax1.yaxis.grid(True, which='major')
    ax1.xaxis.set_tick_params(which='minor', bottom=False)
    ax1.set_xlabel('Month')

    # Ploting Year median and standar deviation
    median_y = []
    for y in years:
        total = df_combined.loc[(df_combined['DATE_TIME'].dt.year == y) & (pd.notna(df_combined[i])), i]
        median_y.append(total)
    ax2 = plt.subplot(1, 3, 2)
    nyear = np.arange(len(years))
    bp2 = ax2.boxplot(median_y, showfliers=False, patch_artist=True)
    for median in bp2['medians']:
        median.set(color='black', linewidth=3)
    for patch in bp2['boxes']:
        patch.set_facecolor(color='white')
    ax2.set_ylabel(plot_lables[j]+units[j])
    ax2.set_xticklabels([str(y)[2:4] for y in years])
    ax2.minorticks_on()
    ax2.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax2.yaxis.grid(True, which='Major')
    ax2.xaxis.set_tick_params(which='minor', bottom=False)
    plt.xticks(rotation=45)
    ax2.set_xlabel('Year')

    # Ploting cumulative histogram
    val_medio = round(df_combined[i].mean(), 1)
    mediana = round(df_combined[i].median(), 1)
    max = round(df_combined[i].max(), 1)
    min = round(df_combined[i].min(), 1)

    c = df_combined.loc[(pd.notna(df_combined[i])), i]
    w = df_combined.loc[(pd.notna(df_combined[i])), "DATE_DIF"]

    ax3 = plt.subplot(1, 3, 3)
    if i == 'WDIR':
        ax3.hist(c, density=True, cumulative=False, bins=NBINS, color="grey", log=False)
    else:
        ax3.hist(c, density=True, cumulative=True, bins=NBINS, color="grey", log=False)
    ax3.set_xlabel(plot_lables[j]+units[j])
    ax3.minorticks_on()
    ax3.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax3.yaxis.grid(True, which='Major')
    ax3.xaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax3.xaxis.grid(True, which='Major')
    if i == 'TEMP':
        ax3.set_xlim([-10, 40])
    if i == 'WSP':
        ax3.set_xlim([0, 40])
    if i == 'HUM':
        ax3.set_xlim([0, 100])
    if i == 'WV':
        ax3.set_xlim([0, 25])

    # ax3.text(0.4,0.55, "Min: "+str(min), transform=ax3.transAxes,fontsize = 12)
    # ax3.text(0.4,0.5,"Max: "+str(max), transform=ax3.transAxes,fontsize = 12)
    # ax3.text(0.4,0.45,"Mean value:  "+str(val_medio), transform=ax3.transAxes,fontsize = 12)
    # ax3.text(0.4,0.4,"Median: "+str(mediana), transform=ax3.transAxes,fontsize = 12)
    plt.tight_layout()
    plt.savefig(OPATH+'/'+i, dpi=300)
    plt.close()
    j = j+1
