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
df_all["water_vapor"] = water_vapor(df_all["Amb_T"],df_all["Hum%"])
df_all["DATE_DIF"] = df_all["DATE"].diff().dt.total_seconds()

for var in COL_NAMES[2:]:
    for i in REMOVE[var]:    
        df_1 = df_all[df_all[var] <= i].index
        df_final = df_all.drop(df_1)
       
print("Creating Images...")

#plot all columns
for variable in COL_NAMES[2:]:
    # vs date
    x = df_final["DATE"]
    y = df_final[variable]
    plt.figure(figsize=[10, 6])
    plt.scatter(x, y, c='b',s=2)
    plt.xlabel(df_final["DATE"].name)
    plt.ylabel(df_final[variable].name)
    plt.tight_layout()
    if variable == "DATE_DIF":
        plt.ylim([0,140])
    plt.savefig(OPATH+'/'+variable, dpi=300)

    #ploting histogram
    plt.figure(figsize=[10, 6])
    if variable == "DATE_DIF":
        plt.hist(y, log=True, bins=NBINS, color='b', histtype='step', range=[0,140])
    else:
        plt.hist(y, log=True, bins=NBINS, color='b', histtype='step')
    plt.grid()
    plt.ylabel('FREQUENCY')
    plt.xlabel(df_final[variable].name)
    plt.tight_layout()
    plt.savefig(OPATH+'/'+variable+'_hist', dpi=300)
    
print("Done")
