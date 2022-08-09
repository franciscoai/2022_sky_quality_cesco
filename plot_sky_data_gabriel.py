# -*- coding: utf-8 -*-
import sys
import logging
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas as pd
from datetime import datetime


REPO_PATH = os.getcwd()
HASTAF = REPO_PATH + '/data/hasta/HASTA_Stats_2009_2020.txt'
MICAF = REPO_PATH + '/data/mica/MICA_Stats_1997_2012.txt'
OPATH = REPO_PATH + '/output/gabriel'  # '/home/gabrielzucarelli/Escritorio/2022_sky_quality_cesco/data/hasta/out_hasta'
REMOVE = {'length': 0.0, 'sunny_time': 0, 'cloudy_time': 0, 'T_ini': 0,
          'T_end': 0, 'sun': 0, 'clou': 0, 'good': 0, 'mod': 0, 'bad': 0}

# HASTA
date_H = []
length = []
sunny_time = []
cloudy_time = []
obs_H = []
flag_H = 1

with open(HASTAF, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')  # archivo en csv
    for row in spamreader:
        if flag_H >= 1 and flag_H <= 3:
            flag_H = flag_H+1
            continue  # permite saltar cada fila hasta que flag>3
        while True:
            try:
                row.remove('')
            except ValueError:
                break

        date_H.append(str(row[0]))
        length.append(row[3])
        length_float = list(np.float_(length))
        sunny_time.append(row[4])
        sunny_time_float = list(np.float_(sunny_time))
        cloudy_time.append(row[5])
        cloudy_time_float = list(np.float_(cloudy_time))
        obs_H.append(row[6])
        obs_H_float = list(np.float_(obs_H))

    date_H = [datetime(int(i[0:4]), int(i[4:6]), int(i[6:8])) for i in date_H]

date_H = np.array(date_H)
length_float = np.array(length_float)
sunny_time_float = np.array(sunny_time_float)
cloudy_time_float = np.array(cloudy_time_float)
obs_H_float = np.array(obs_H_float)

# MICA
date_M = []
T_ini = []
T_end = []
sun = []
clou = []
good = []
mod = []
bad = []
obs_M = []
flag_M = 1

with open(MICAF, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')  # archivo en csv
    for row in spamreader:
        if flag_M < 2:
            flag_M = flag_M + 1
            continue
        while True:
            try:
                row.remove('')
            except ValueError:
                break

        date_M.append(str(row[0]))
        T_ini.append(row[1])
        T_ini_float = list(np.float_(T_ini))
        T_end.append(row[2])
        T_end_float = list(np.float_(T_end))
        sun.append(row[3])
        sun_float = list(np.float_(sun))
        clou.append(row[4])
        clou_float = list(np.float_(clou))
        good.append(row[5])
        good_float = list(np.float_(good))
        mod.append(row[6])
        mod_float = list(np.float_(mod))
        bad.append(row[7])
        bad_float = list(np.float_(bad))
        obs_M.append(row[8])
        obs_M_float = list(np.float_(obs_M))

    date_M = [datetime(int(i[0:4]), int(i[4:6]), int(i[6:8])) for i in date_M]

date_M = np.array(date_M)
T_ini_float = np.array(T_ini_float)
T_end_float = np.array(T_end_float)
sun_float = np.array(sun_float)
clou_float = np.array(clou_float)
good_float = np.array(good_float)
mod_float = np.array(mod_float)
bad_float = np.array(bad_float)
obs_M_float = np.array(obs_M_float)

# HASTA
# length_float
fig = plt.figure(figsize=(10, 5))
okind = np.argwhere(length_float != REMOVE['length'])
plt.scatter(date_H[okind], length_float[okind], s=5)
plt.xlabel("Date")
plt.ylabel("length [h]")
plt.tight_layout()
plt.savefig(OPATH+'/length_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(length_float[okind])
plt.xlabel('length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/length_float_hist')
plt.close()

# sunny_time_float
fig = plt.figure(figsize=(10, 5))
okind = np.argwhere(sunny_time_float != REMOVE['sunny_time'])
plt.scatter(date_H[okind], sunny_time_float[okind], s=5)
plt.xlabel("Date")
plt.ylabel("Sunny time [h]")
plt.tight_layout()
plt.savefig(OPATH+'/sunny_time_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(sunny_time_float[okind])
plt.xlabel('Sunny Time [h]')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/sunny_time_float_hist')
plt.close()

# cloudy_time_float
fig = plt.figure(figsize=(10, 5))
okind = np.argwhere(cloudy_time_float != REMOVE['cloudy_time'])
plt.scatter(date_H[okind], cloudy_time_float[okind], s=5)
plt.xlabel("Date")
plt.ylabel("cloudy time")
plt.tight_layout()
plt.savefig(OPATH+'/cloudy_time_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(sunny_time_float[okind])
plt.xlabel('cloudy time')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/cloudy_time_float_hist')
plt.close()

# obs_H_float
fig = plt.figure(figsize=(10, 5))
plt.scatter(date_H, obs_H_float, s=5)
plt.xlabel("Date")
plt.ylabel("obs(H)")
plt.tight_layout()
plt.savefig(OPATH+'/obs_H_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(obs_H_float)
plt.xlabel('obs H')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/obs_H_float_hist')
plt.close()


# MICA

# T_ini_float
fig = plt.figure(figsize=(10, 5))
okind = np.argwhere(T_ini_float != REMOVE['T_ini'])
plt.scatter(date_M[okind], T_ini_float[okind], s=5)
plt.xlabel("Date")
plt.ylabel("T_ini")
plt.tight_layout()
plt.savefig(OPATH+'/T_ini_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(T_end_float[okind])
plt.xlabel('T_ini')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/T_ini_float_hist')
plt.close()

# T_end_float
fig = plt.figure(figsize=(10, 5))
okind = np.argwhere(T_end_float != REMOVE['T_end'])
plt.scatter(date_M[okind], T_end_float[okind], s=5)
plt.xlabel("Date")
plt.ylabel("T_end")
plt.tight_layout()
plt.savefig(OPATH+'/T_end_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(T_end_float[okind])
plt.xlabel('T_end')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/T_end_float_hist')
plt.close()

# sun_float
fig = plt.figure(figsize=(10, 5))
okind = np.argwhere(sun_float != REMOVE['sun'])
plt.scatter(date_M[okind], sun_float[okind], s=5)
plt.xlabel("Date")
plt.ylabel("sun")
plt.tight_layout()
plt.savefig(OPATH+'/sun_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(sun_float[okind])
plt.xlabel('sun_float')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/sun_float_hist')
plt.close()

# clou_float
fig = plt.figure(figsize=(10, 5))
okind = np.argwhere(clou_float != REMOVE['clou'])
plt.scatter(date_M[okind], clou_float[okind], s=5)
plt.xlabel("Date")
plt.ylabel("clou")
plt.tight_layout()
plt.savefig(OPATH+'/clou_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(clou_float[okind])
plt.xlabel('clou')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/clou_float_hist')
plt.close()

# good_float

fig = plt.figure(figsize=(10, 5))
okind = np.argwhere(good_float != REMOVE['good'])
plt.scatter(date_M[okind], good_float[okind], s=5)
plt.xlabel("Date")
plt.ylabel("good")
plt.tight_layout()
plt.savefig(OPATH+'/good_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(good_float[okind])
plt.xlabel('good')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/good_float_hist')
plt.close()

# mod_float
fig = plt.figure(figsize=(10, 5))
okind = np.argwhere(mod_float != REMOVE['mod'])
plt.scatter(date_M[okind], mod_float[okind], s=5)
plt.xlabel("Date")
plt.ylabel("mod")
plt.tight_layout()
plt.savefig(OPATH+'/mod_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(mod_float[okind])
plt.xlabel('mod')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/mod_float_hist')
plt.close()

# bad_float
fig = plt.figure(figsize=(10, 5))
okind = np.argwhere(bad_float != REMOVE['bad'])
plt.scatter(date_M[okind], bad_float[okind], s=5)
plt.xlabel("Date")
plt.ylabel("bad_float")
plt.tight_layout()
plt.savefig(OPATH+'/bad_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(bad_float[okind])
plt.xlabel('bad')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/bad_float_hist')
plt.close()

# obs_M_float
fig = plt.figure(figsize=(10, 5))
plt.scatter(date_M, obs_M_float, s=5)
plt.xlabel("Date")
plt.ylabel("obs_M")
plt.tight_layout()
plt.savefig(OPATH+'/obs_M_float_vs_date.png')
plt.close()

fig = plt.figure(figsize=(10, 5))
plt.hist(obs_M_float)
plt.xlabel('obs_M')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OPATH+'/obs_M_float_hist')
plt.close()
