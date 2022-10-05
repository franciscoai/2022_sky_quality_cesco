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
import matplotlib.dates as mdates
import matplotlib.colors as colors
from scipy.optimize import curve_fit
from water_vapor import water_vapor
import pysolar.solar as solar
import fitsio


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


REPO_PATH = os.getcwd()
MICAF = None
MICA_DIRS = ['/media/sf_iglesias_data/cesco_sky_quality/MICA_processed/16_nov_2005/Treatimg',
             '/media/sf_iglesias_data/cesco_sky_quality/MICA_processed/22_ago_2006/Treatimg', ]
OPATH = REPO_PATH + '/output/mica_vs_tester'
SCIFMT = '{:6.2f}'

# read the headers of MICA processed images
mf = []
for d in MICA_DIRS:
    mf = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('tgr.fts')]

mh = []
for f in mf:
    mh.append(fitsio.read_header(f))

sky12 = []
sky1 = []
sky2 = []
skyt = []
sunt = []

for h in mh:
    sky12.append(h['SKY1.2'])  # in ppm
    sky1.append(h['SKYSUN'])  # in ppm
    sky2.append(fit_func(250, h['SKYFIT0'], h['SKYFIT1'], h['SKYFIT2']))  # in ppm
    skyt.append(h['SKYTEST'])  # in V
    sunt.append(h['SUNTEST'])  # in V
sky12 = np.array(sky12)
sky1 = np.array(sky1)
sky2 = np.array(sky2)
skyt = np.array(skyt)
sunt = np.array(sunt)

# Plots
print('Plotting '+str(len(mh))+' files')
os.makedirs(OPATH, exist_ok=True)

# scatter
ok_ind = np.where(sky12 > 1)
y = sky12[ok_ind]
x = skyt[ok_ind]/sunt[ok_ind]
[m, b], residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
chisq_dof = residuals / (len(x))
print(chisq_dof)
plt. plot(x, y, 'o', label='Measurements')
plt. plot(x, m*x + b, label='$I_{Mica(1.2)}$='+SCIFMT.format(m)+'$*I_{Tester}$ + ' +
          SCIFMT.format(b) + ' ; rms residual='+SCIFMT.format(chisq_dof[0]) + 'ppm')
plt.ylabel('$I_{Mica(1.2)}$[ppm]')
plt.xlabel('$I_{Tester}$')
plt.legend()
plt.savefig(OPATH+'/tester_mica12.png')
plt.close()

ok_ind = np.where(sky1 > 1)
y = sky1[ok_ind]
x = skyt[ok_ind]/sunt[ok_ind]
[m, b], residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
chisq_dof = residuals / (len(x))
print(chisq_dof)
plt. plot(x, y, 'o', label='Measurements')
plt. plot(x, m*x + b, label='$I_{Mica(1)}$='+SCIFMT.format(m)+'$*I_{Tester}$ + ' +
          SCIFMT.format(b) + ' ; rms residual='+SCIFMT.format(chisq_dof[0]) + 'ppm')
plt.ylabel('$I_{Mica(1)}$[ppm]')
plt.xlabel('$I_{Tester}$')
plt.legend()
plt.savefig(OPATH+'/tester_mica1.png')
plt.close()

ok_ind = np.where(sky2 > 1)
y = sky2[ok_ind]
x = skyt[ok_ind]/sunt[ok_ind]
[m, b], residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
chisq_dof = residuals / (len(x))
print(chisq_dof)
plt. plot(x, y, 'o', label='Measurements')
plt. plot(x, m*x + b, label='$I_{Mica(2)}$='+SCIFMT.format(m)+'$*I_{Tester}$ + ' +
          SCIFMT.format(b) + ' ; rms residual='+SCIFMT.format(chisq_dof[0]) + 'ppm')
plt.ylabel('$I_{Mica(2)}$[ppm]')
plt.xlabel('$I_{Tester}$')
plt.legend()
plt.savefig(OPATH+'/tester_mica2.png')
plt.close()

# # radial profile
# i = 0
# for f in mf:
#     print(f)
#     data = fitsio.read(f)
#     cenx = mh[i]['HOLE0']
#     ceny = mh[i]['HOLE1']
#     [x, y] = radial_avg(data, cenx, ceny)
#     rng = slice(0, 1023)
#     # plt.plot(x[rng],y[rng]*mh[i]['CT2PPM'],'-*r')
#     mb12 = fit_func(50, mh[i]['SKYFIT0'], mh[i]['SKYFIT1'], mh[i]['SKYFIT2'])
#     mblimb = fit_func(0, mh[i]['SKYFIT0'], mh[i]['SKYFIT1'], mh[i]['SKYFIT2'])
#     plt.plot(x[rng], fit_func(x[rng], mh[i]['SKYFIT0'], mh[i]['SKYFIT1'], mh[i]['SKYFIT2']),
#              '-*b', label='50px ='+SCIFMT.format(mb12)+' ; 0 px ='+SCIFMT.format(mblimb))
#     plt.plot([rng.start, rng.stop], [mh[i]['SKY1.2'], mh[i]['SKY1.2']], '-g',
#              label='SKY1.2='+SCIFMT.format(mh[i]['SKY1.2'])+' ; SKYSUN='+SCIFMT.format(mh[i]['SKYSUN']))
#     plt.legend()
#     plt.show()
#     i += 1
