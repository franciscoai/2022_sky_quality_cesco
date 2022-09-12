from calendar import month
from pickle import TRUE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from water_vapor import water_vapor

REPO_PATH = os.getcwd()
path = REPO_PATH + '/data/master'
OPATH = REPO_PATH + '/output/master'
DAY_TIME=[datetime.time(hour=10,minute=0),datetime.time(hour=22,minute=0)] # daytime interval
COL_NAMES = ["DAY","HOUR(UT)","Sky_T","Amb_T","WindS","Hum%","DewP","C", "W", "R", "Cloud_T","DATE_DIF","water_vapor"]  
NBINS = 50
REMOVE =  {"Sky_T":[-990],"Amb_T":[],"WindS":[],"Hum%":[],"DewP":[],"C":[], "W":[], "R":[], "Cloud_T": [],"DATE_DIF":[],"water_vapor":[]}

# create opath
os.makedirs(OPATH, exist_ok=True)
mf = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_wea.txt')]

#reading all files and converting to datetime
df_all = []
for f in mf:
    df = pd.read_csv(f, delim_whitespace=True, skiprows=1, names=COL_NAMES)
    df["DAY"] = pd.to_datetime(df["DAY"], format="%Y%m%d")
    df["HOUR(UT)"] = [datetime.timedelta(hours=h) for h in df['HOUR(UT)']]
    df["DATE"] = df["DAY"]+df["HOUR(UT)"]
    df_all.append(df)
df_all = pd.concat(df_all, ignore_index=True)

df_all["Cloud_T"] =  df_all["Sky_T"] - df_all["Amb_T"]
df_all["DATE_DIF"] = df_all["DATE"].diff().dt.total_seconds()

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
    

#reading weather data
COL_NAMES_WEA =["DATE","HOUR(UT)","TEMP","PRESS","HUM","WSP","WDIR"]
#weather path
REPO_WEA_PATH = os.getcwd()
path_wea = REPO_WEA_PATH + '/data/wea'
OPATH_WEA = REPO_WEA_PATH + '/output/wea'

# create opath
os.makedirs(OPATH_WEA, exist_ok=True)
mf_wea = [os.path.join(path_wea, f) for f in os.listdir(path_wea) if f.endswith('.txt')]

#reading all files and converting to datetime
df_all_wea = []
for g in mf_wea:
    df_wea = pd.read_csv(g, delim_whitespace=True, skiprows=2, names=COL_NAMES_WEA,encoding='latin1')
    df_wea["DATE"] = pd.to_datetime(df_wea["DATE"], format="%Y%m%d")
    df_wea["HOUR(UT)"] =pd.to_timedelta(df_wea["HOUR(UT)"])
    df_wea["DATE_TIME"] = df_wea["DATE"]+df_wea["HOUR(UT)"]
    df_wea['date_diff'] = df_wea['DATE_TIME'].diff().dt.total_seconds()
    df_all_wea.append(df_wea)
df_all_wea = pd.concat(df_all_wea, ignore_index=True)

df_all_wea["water_vapor"] = water_vapor(df_all["Amb_T"],df_all["Hum%"])

# creating df with day hours and droping rows with inconclusive values
df_weather = df_all_wea.loc[(df_all_wea["DATE_TIME"].dt.hour> 9) & (df_all_wea["DATE_TIME"].dt.hour<22) ]
df_weather.drop(df_weather[df_weather['TEMP'] <= -50].index, inplace = True)
df_weather.drop(df_weather[df_weather['TEMP'] >= 50].index, inplace = True)

months = df_weather['DATE_TIME'].dt.month.unique()
years = df_weather['DATE_TIME'].dt.year.unique()
variables=["TEMP","PRESS","HUM","WSP","WDIR", "water_vapor"]

#Ploting graphics
for i in variables:
    fig1, (ax1,ax2,ax3) = plt.subplots(3)
    fig1.set_size_inches(12, 4)
    #ploting seasonal median and standar deviation
    median=[]
    std=[]
    for m in months:
        tot = df_weather.loc[df_weather['DATE_TIME'].dt.month == m,i]
        median.append(tot.mean())
        std.append(tot.std())
    ax1 = plt.subplot(1,3,1)
    ax1.bar(months, median, yerr=std, align='center', alpha=0.5,color="grey", ecolor='black', capsize=10)
    ax1.plot(months, median,"ko")
    ax1.set_ylabel(i)
    ax1.set_xticks(months)
    ax1.set_xticklabels(months)
    ax1.set_title('Seasonal median and standar deviation')

    #Ploting Year median and standar deviation
    median_y=[]
    std_y=[]
    for y in years:
        total = df_weather.loc[df_weather['DATE_TIME'].dt.year == y,i]
        median_y.append(total.median())
        std_y.append(total.std())
    ax2 = plt.subplot(1,3,2)
    ax2.bar(years, median_y, yerr=std_y, align='center', alpha=0.5,color="grey", ecolor='black', capsize=10)
    ax2.plot(years, median_y,"ko")
    ax2.set_ylabel(i)
    ax2.set_xticks(years)
    ax2.set_xticklabels(years, rotation=45)
    ax2.set_title('Years median and standar deviation')
    ax2.yaxis.grid(True)
    
    #Ploting cumulative histogram
    ax3 = plt.subplot(1,3,3)
    ax3.hist(df_weather[i], weights=df_weather["date_diff"], cumulative=TRUE, color="grey")
    ax3.set_xlabel(i)
    ax3.set_ylabel("%")
    ax3.set_title('Cumulative Distribution')
    plt.tight_layout()
    plt.show()
    #plt.savefig(OPATH+'/'+i, dpi=300)