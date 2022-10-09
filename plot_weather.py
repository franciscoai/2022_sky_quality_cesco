from calendar import month
from pickle import TRUE
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
NBINS = 20
REMOVE = {"Sky_T": [-990], "Amb_T": [], "WindS": [], "Hum%": [], "DewP": [],
          "C": [], "W": [], "R": [], "Cloud_T": [], "DATE_DIF": [], "WV": []}
variables = ["TEMP", "PRESS", "HUM", "WSP", "WDIR", "WV", "date_diff"]  # variables to plot
units = ["  [$^\circ C$]", "  [mm Hg]","  [%]","  [m$s^{-1}$]","  [Deg]","  [mm]",""]
remove_weather_min = {"TEMP":[-20], "PRESS": [], "HUM": [0], "WSP": [0], "WDIR": [], "WV": [], "date_diff": [0]}
remove_weather_max = {"TEMP":[40], "PRESS": [], "HUM": [150], "WSP": [40], "WDIR": [], "WV": [], "date_diff": [3600]}

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



# print("Creating Images...")

# #plot all columns
# for variable in COL_NAMES[2:]:
#     # vs date
#     x = df_final["DATE"]
#     y = df_final[variable]
#     plt.figure(figsize=[10, 6])
#     plt.scatter(x, y, c='b',s=2)
#     plt.xlabel(df_final["DATE"].name)
#     plt.ylabel(df_final[variable].name)
#     plt.tight_layout()
#     if variable == "DATE_DIF":
#         plt.ylim([0,140])
#     plt.savefig(OPATH+'/'+variable, dpi=300)

#     #ploting histogram
#     plt.figure(figsize=[10, 6])
#     if variable == "DATE_DIF":
#         plt.hist(y, log=True, bins=NBINS, color='b', histtype='step', range=[0,140])
#     else:
#         )plt.hist(y, log=True, bins=NBINS, color='b', histtype='step')
#     plt.grid()
#     plt.ylabel('FREQUENCY')
#     plt.xlabel(df_final[variable].name)
#     plt.tight_layout()
#     plt.savefig(OPATH+'/'+variable+'_hist', dpi=300)


# reading weather data
mf_wea = [os.path.join(path_wea, f) for f in os.listdir(path_wea) if f.endswith('.txt')]

# reading all files and converting to datetime
df_all_wea = []
for g in mf_wea:
    df_wea = pd.read_csv(g, delim_whitespace=True, skiprows=2, names=COL_NAMES_WEA, encoding='latin1')
    df_wea["DATE"] = pd.to_datetime(df_wea["DATE"], format="%Y%m%d")
    df_wea["HOUR(UT)"] = pd.to_timedelta(df_wea["HOUR(UT)"])
    df_wea["DATE_TIME"] = df_wea["DATE"]+df_wea["HOUR(UT)"]
    df_wea['date_diff'] = df_wea['DATE_TIME'].diff().dt.total_seconds()
    df_all_wea.append(df_wea)
df_all_wea = pd.concat(df_all_wea, ignore_index=True)
df_all_wea["WV"] = water_vapor(df_all["Amb_T"], df_all["Hum%"])

# creating df with day hours
df_weather = df_all_wea.loc[(df_all_wea["DATE_TIME"].dt.hour > 9) & (df_all_wea["DATE_TIME"].dt.hour < 22)]

# remove values
for var in variables:
    for i in remove_weather_min[var]:
        df_2 = df_weather[df_weather[var] <= i].index
        df_weather = df_weather.drop(df_2)

for var in variables:
    for i in remove_weather_max[var]:
        df_3 = df_weather[df_weather[var] >= i].index
        df_weather= df_weather.drop(df_3)

months = df_weather['DATE_TIME'].dt.month.unique()
years = np.sort(df_weather['DATE_TIME'].dt.year.unique())

j = 0
# Ploting graphics
for i in variables:
    fig1, (ax1, ax2, ax3) = plt.subplots(3)
    fig1.set_size_inches(13, 4)
    median = []
    colors = '#e4e8f0'
    for m in months:
        tot = df_weather.loc[df_weather['DATE_TIME'].dt.month == m, i]
        median.append(tot)
    ax1 = plt.subplot(1, 3, 1)
    bp = ax1.boxplot(median, showfliers=False, patch_artist = True)
    for median in bp['medians']:
        median.set(color ='black',linewidth = 3)
    for patch in bp['boxes']:
        patch.set_facecolor(color = 'white')
    ax1.set_ylabel(i+units[j])
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
        total = df_weather.loc[df_weather['DATE_TIME'].dt.year == y, i]
        median_y.append(total)
    ax2 = plt.subplot(1, 3, 2)
    nyear = np.arange(len(years))
    bp2 = ax2.boxplot(median_y, showfliers=False, patch_artist = True)
    for median in bp2['medians']:
        median.set(color ='black',linewidth = 3)
    for patch in bp2['boxes']:
        patch.set_facecolor(color = 'white')  
    ax2.set_ylabel(i+units[j])
    ax2.set_xlabel('Year from 2000')
    ax2.set_xticks(nyear[::2])
    ax2.minorticks_on()
    ax2.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax2.yaxis.grid(True, which='Major')
    ax2.xaxis.set_tick_params(which='minor', bottom=False)

    # Ploting cumulative histogram

    val_medio=round(df_weather[i].mean(),3)
    mediana=df_weather[i].median()
    max= df_weather[i].max()
    min=df_weather[i].min()
    ax3 = plt.subplot(1, 3, 3)
    ax3.hist(df_weather[i], weights=df_weather["date_diff"], density=True,
             cumulative=True, color="grey", bins=NBINS, log=True)
    ax3.set_xlabel(i+units[j])
    ax3.minorticks_on()
    ax3.yaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax3.yaxis.grid(True, which='Major')
    ax3.xaxis.grid(which="minor", linestyle=':', linewidth=0.7)
    ax3.xaxis.grid(True, which='Major')
    ax3.text(5,0.01,"min: "+str(min))
    ax3.text(5,0.006,"max: "+str(max))
    ax3.text(5,0.004,"medium value:  "+str(val_medio))
    ax3.text(5,0.0025,"median: "+str(mediana))
    plt.tight_layout()
    plt.savefig(OPATH+'/'+i, dpi=300)
    plt.close()
    j=j+1
