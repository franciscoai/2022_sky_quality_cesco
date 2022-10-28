# -*- coding: utf-8 -*-
from cProfile import label
import pickle
from lib2to3.pgen2.token import OP
import os
from textwrap import shorten
from turtle import clear, color
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta
import copy
from pyparsing import col
import matplotlib.dates as mdates
import matplotlib.colors as colors
from scipy.optimize import curve_fit
from water_vapor import water_vapor
import pysolar.solar as solar
import fitsio
from zipfile import ZipFile


def fit_func(x, SKYFIT0, SKYFIT1, SKYFIT2):
    # to compute sky brigthness [ppm]. Taken from MICA the header
    if SKYFIT2 != 0:
        return SKYFIT0 + SKYFIT1*np.exp(x/SKYFIT2)
    else:
        return np.nan


def radial_avg(img, cenx, ceny):
    '''
    Returns the radial average of the input image around the center  [cenx,ceny]
    '''
    # Get image parameters
    a = img.shape[0]
    b = img.shape[1]
    # Find radial distances
    [X, Y] = np.meshgrid(np.arange(b) - cenx, np.arange(a) - ceny)
    R = np.sqrt(np.square(X) + np.square(Y))
    rad = np.arange(1, np.max(R), 1)
    intensity = np.zeros(len(rad))
    index = 0
    bin_size = 1
    for i in rad:
        mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
        values = img[mask]
        intensity[index] = np.mean(values)
        index += 1
    return ([rad, intensity])

# CONSTANTS


REPO_PATH = os.getcwd()
MICAF = None
MICA_DIRS = '/media/sf_iglesias_data/cesco_sky_quality/MICA_processed'
OPATH = REPO_PATH + '/output/mica_calibration'
SCIFMT = '{:6.2f}'
FILE_TYPE = 'x26'  # 'tgr' # 'x26'  # file ending to read

#####################
# read the headers of MICA processed images
dirs = [os.path.join(MICA_DIRS, f) for f in os.listdir(MICA_DIRS) if (f.startswith('X2_') or f.startswith('Tgr_'))]
mf = []
for d in dirs:
    mf.append([os.path.join(d, f) for f in os.listdir(d) if f.endswith(FILE_TYPE+'.fts')])
mh = []
i = 0
for d in mf:
    print('Reading dir '+str(i)+' of '+str(len(mf)))
    i += 1
    for f in d:
        mh.append(fitsio.read_header(f))

sky12 = []
sky1 = []
sky2 = []
skyt = []
sunt = []
sky6 = []
sky8 = []
date = []
for h in mh:
    sky12.append(h['SKY1.2'])  # in ppm
    sky1.append(h['SKYSUN'])  # in ppm
    sky2.append(fit_func(250, h['SKYFIT0'], h['SKYFIT1'], h['SKYFIT2']))  # in ppm
    skyt.append(h['SKYTEST'])  # in V
    sunt.append(h['SUNTEST'])  # in V
    sky6.append(fit_func(250*3, h['SKYFIT0'], h['SKYFIT1'], h['SKYFIT2']))  # in ppm
    sky8.append(fit_func(250*4, h['SKYFIT0'], h['SKYFIT1'], h['SKYFIT2']))  # in ppm
    date.append(datetime.strptime(h['DATE-OBS'], '%Y/%m/%d'))
    if (h['FILTER'] != 'Fe-XIV L') & (h['FILTER'] != 'Fe-XIV C'):
        print('Error, the following file has a diff FILTER value: ' + h['FILENAME'])
sky12 = np.array(sky12)
sky1 = np.array(sky1)
sky2 = np.array(sky2)
skyt = np.array(skyt)
sunt = np.array(sunt)
sky6 = np.array(sky6)
sky8 = np.array(sky8)
date = np.array(date)
df = pd.DataFrame({'date': date, 'sunt': sunt, 'skyt': skyt, 'Itester': skyt/sunt, 'sky1': sky1,
                  'sky12': sky12, 'sky2': sky2, 'sky6': sky6, 'sky8': sky8})

# Plots
print('Plotting '+str(len(mh))+' Mica measurements')
os.makedirs(OPATH, exist_ok=True)

# scatter
# use only good sky conditions !!!!
df1 = copy.deepcopy(df.loc[(df['sky6'] > 0) & (df['sky6'] < 40)])
x1 = df1['Itester']
y1 = df1['sky6']
[m, b], residuals, _, _, _ = np.polyfit(x1, y1, 1, full=True)
df1['residuals'] = pd.Series.abs(m*x1 + b - y1)
# deletes extreme values
print('Quantile '+str(df1['residuals'].quantile(.99)))
if FILE_TYPE == 'x26':
    df2 = df1
    mksize = 5
else:
    df2 = df1.loc[df1['residuals'] < df1['residuals'].quantile(.99)]
    mksize = 2.5
plt.plot(df['Itester'], df['sky6'], '.', label='Measurement not used in the fit', color='grey', markersize=mksize)
print('valid data covers '+str(len(df2['date'].dt.date.unique()))+' days')
print(df2['date'].dt.date.unique())
x=df2['Itester']
y=df2['sky6']
[m, b], residuals, _, _, _=np.polyfit(x, y, 1, full=True)
chisq_dof=residuals / (len(x))
print('rms residual=')
for i in np.sort(df2['date'].dt.month.unique()):
    df3=df2.loc[(df2['date'].dt.month == i)]
    # , label=str(df3['date'].dt.year.unique())[1:-1]+'/'+str(i)
    plt.plot(df3['Itester'], df3['sky6'], '.', markersize=mksize, color='k')
plt.plot(df3['Itester'], df3['sky6'], '.', markersize=mksize, color='k', label='Measurement used in the fit')
plt.plot(x, m*x + b, '-', color='grey', markersize=mksize, label='$I_{mica}$='+SCIFMT.format(m)+'$*\dfrac{Sky-T}{Sun-T}$ + ' +
         SCIFMT.format(b)+'$ \pm$'+SCIFMT.format(chisq_dof[0]))

# # fits only the best conditions values
# df2 = df2.loc[df2['Itester'] < 0.2]
# x = df2['Itester']
# y = df2['sky6']
# [m, b], residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
# chisq_dof = residuals / (len(x))
# plt.plot(x, m*x + b, '--y', label='$I_{Mica}$='+SCIFMT.format(m)+'$*I_{Tester<0.2}$ + ' +
#          SCIFMT.format(b)+'$\pm$'+SCIFMT.format(chisq_dof[0]))

plt.ylim([0,60])
plt.ylabel('$I_{mica}$ [ppm]')
plt.xlabel('Sky-T/Sun-T')
plt.grid('both')
plt.tight_layout()
plt.legend(loc='best', prop={'size': 9})
plt.savefig(OPATH+'/tester_mica6'+FILE_TYPE+'.png')
plt.close()

# Plot various linear fits
fits=[['tgr', 70.35, 9.77], ['x1c', 87.95, 7.16], ['x26', 75.62, 8.13]]  # at 2 Rs
x=np.arange(0, 1, 0.05)
for f in fits:
    plt.plot(x, f[1]*x+f[2], label=f[0])
plt.xlabel('Sky-T/Sun_T')
plt.ylabel('$I_{Mica}$')
plt.legend()
plt.grid('both')
plt.savefig(OPATH+'/all_fits.png')
plt.close()
