# -*- coding: utf-8 -*-
import sys
import logging
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import csv

__author__ = "Francisco Iglesias"
__copyright__ = "Copyright 2021"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"

"""
Plots sky conditions summary data from Carlos Francille
"""
REPO_PATH = os.getcwd()
HASTAF = REPO_PATH + '/data/hasta/HASTA_Stats_summary.txt'
MICAF = REPO_PATH + '/data/mica/MICA_Stats_summary.txt'
OPATH = REPO_PATH + '/output'
NOYH = 12  # number of years
NOYM = 16
STRFMTS = '{:4.1f}'  # format string to print short decimal numbers
BWIDTH = 0.45
DPI = 300.  # image dpi
FIGSZ = 15, 9  # immage size in inches
IMGFMT = '.png'  # '.eps, .pdf' # Image format of the output plots

# reads files
h = ["bla"]
with open(HASTAF, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        if row[0].find('#'):
            while("" in row):
                row.remove("")
            h.append(row)
h = h[1:]

m = ["bla"]
with open(MICAF, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        if row[0].find('#'):
            while("" in row):
                row.remove("")
            m.append(row)
m = m[1:]

# output dir
os.makedirs(OPATH, exist_ok=True)

# plots days per year
plt.subplots(1, 1, figsize=FIGSZ)
# hasta
y = np.array([float(h[i][0]) for i in range(NOYH)])
y[0:4] += 0.5
d = np.array([float(h[i][1]) for i in range(NOYH)])/3.65
d1 = np.array([float(h[i][3]) for i in range(NOYH)])/3.65
d2 = np.array([float(h[i][5]) for i in range(NOYH)])/3.65
plt.bar(y, d, BWIDTH, color='g', zorder=3, alpha=0.6)
plt.bar(y, d2, BWIDTH, color='k', bottom=d, zorder=3, alpha=0.6)
plt.bar(y, d1, BWIDTH, color='r', bottom=d+d2, zorder=3, alpha=0.6)
# mica
y = np.array([float(m[i][0]) for i in range(NOYM)])
dm = np.array([float(m[i][1]) for i in range(NOYM)])/3.65
d1m = np.array([float(m[i][3]) for i in range(NOYM)])/3.65
d2m = np.array([float(m[i][5]) for i in range(NOYM)])/3.65
lbl = "sunny (M) ; observed (H)=" + STRFMTS.format(np.mean(dm[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(dm[1:-1])) + " ; "+STRFMTS.format(np.mean(d[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d[1:-1]))+"%"
plt.bar(y, dm, BWIDTH, color='g', label=lbl, zorder=3)
lbl = "not observed(M;H)="+STRFMTS.format(np.mean(d2m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d2m[1:-1])) + " ; "+STRFMTS.format(np.mean(d2[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d2[1:-1]))+"%"
plt.bar(y, d2m, BWIDTH, color='k', label=lbl, bottom=dm, zorder=3)
lbl = "fully cloudy(M;H)=" + STRFMTS.format(np.mean(d1m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d1m[1:-1])) + " ; "+STRFMTS.format(np.mean(d1[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d1[1:-1]))+"%"
plt.bar(y, d1m, BWIDTH, color='r', label=lbl, bottom=dm+d2m, zorder=3)
plt.title("CESCO (OAFA) sky conditions from MICA (1997-2012) and HASTA (2009-2020) data \n" +
          "(avg $\pm$ std excluding: M1997, M2012, H2009, H2020)")
plt.yticks(np.arange(0, 110, 10))
plt.xticks(np.arange(1997, 2021, 1))
plt.legend(loc="lower right")
plt.grid(zorder=0)
plt.ylabel("day/365 [%]")
plt.xlabel("year")
plt.savefig(OPATH+'/TABLE1_days_year'+IMGFMT, dpi=DPI)

# plots hours per year / sunny interval
plt.subplots(1, 1, figsize=FIGSZ)
# hasta
y = np.array([float(h[i+NOYH][0]) for i in range(NOYH)])
y[0:4] += 0.5
d = np.array([float(h[i+NOYH][2]) for i in range(NOYH)])
d1 = np.array([float(h[i+NOYH][4]) for i in range(NOYH)])
d2 = np.array([float(h[i+NOYH][6]) for i in range(NOYH)])
#plt.bar(y,d,BWIDTH, color='g', label="observed",zorder=3)
plt.bar(y, d1, BWIDTH, color='g', bottom=0, zorder=3, alpha=0.6)
plt.bar(y, d2, BWIDTH, color='r', bottom=d1, zorder=3, alpha=0.6)
# mica
y = np.array([float(m[i+NOYM][0]) for i in range(NOYM)])
dm = np.array([float(m[i+NOYM][2]) for i in range(NOYM)])
d1m = np.array([float(m[i+NOYM][4]) for i in range(NOYM)])+dm  # good plus moderate
d2m = np.array([float(m[i+NOYM][6]) for i in range(NOYM)])
#plt.bar(y,dm,BWIDTH, color='g',zorder=3)
#lbl="observed (M/H)="+ STRFMTS.format(np.mean(dm[1:-1]))+"$\pm$"+STRFMTS.format(np.std(dm[1:-1])) +" / "+STRFMTS.format(np.mean(d[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d[1:-1]))+"%"
#plt.bar(y,dm,BWIDTH, color='g', label=lbl,zorder=3)
lbl = "good + moderate (M) ; good (H)=" + STRFMTS.format(np.mean(d1m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d1m[1:-1])) + " ; "+STRFMTS.format(np.mean(d1[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d1[1:-1]))+"%"
plt.bar(y, d1m, BWIDTH, color='g', label=lbl, bottom=0, zorder=3)
lbl = "bad (M) ; with clouds (H)="+STRFMTS.format(np.mean(d2m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d2m[1:-1])) + " ; "+STRFMTS.format(np.mean(d2[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d2[1:-1]))+"%"
plt.bar(y, d2m, BWIDTH, color='r', label=lbl, bottom=d1m, zorder=3)
plt.title("CESCO (OAFA) sky conditions from MICA (1997-2012) and HASTA (2009-2020) data \n" +
          "(avg $\pm$ std excluding: M1997, M2012, H2009, H2020)")
#plt.yticks(np.arange(0, 110, 10))
plt.xticks(np.arange(1997, 2021, 1))
plt.legend(loc="lower right")
plt.grid(zorder=0)
plt.ylabel("hour / sunny obs. interval [%]")
plt.xlabel("year")
plt.savefig(OPATH+'/TABLE2_hours_year_sunny_interval'+IMGFMT, dpi=DPI)

# plots hours per year / default interval
plt.subplots(1, 1, figsize=FIGSZ)
# hasta
y = np.array([float(h[i+2*NOYH][0]) for i in range(NOYH)])
y[0:4] += 0.5
d = np.array([float(h[i+2*NOYH][2]) for i in range(NOYH)])
d1 = np.array([float(h[i+2*NOYH][4]) for i in range(NOYH)])
d2 = np.array([float(h[i+2*NOYH][6]) for i in range(NOYH)])
#plt.bar(y,d,BWIDTH, color='g', label="observed",zorder=3)
plt.bar(y, d1, BWIDTH, color='g', bottom=0, zorder=3, alpha=0.6)
plt.bar(y, d2, BWIDTH, color='r', bottom=d1, zorder=3, alpha=0.6)
# mica
y = np.array([float(m[i+2*NOYM][0]) for i in range(NOYM)])
dm = np.array([float(m[i+2*NOYM][2]) for i in range(NOYM)])
d1m = np.array([float(m[i+2*NOYM][4]) for i in range(NOYM)])
d2m = np.array([float(m[i+2*NOYM][6]) for i in range(NOYM)])
#plt.bar(y,dm,BWIDTH, color='g',zorder=3)
lbl = "sunny (M); good (H)=" + STRFMTS.format(np.mean(d1m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d1m[1:-1])) + " ; "+STRFMTS.format(np.mean(d1[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d1[1:-1]))+"%"
plt.bar(y, d1m, BWIDTH, color='g', label=lbl, bottom=0, zorder=3)
lbl = "cloudy (M); with clouds (H)="+STRFMTS.format(np.mean(d2m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d2m[1:-1])) + " ; "+STRFMTS.format(np.mean(d2[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d2[1:-1]))+"%"
plt.bar(y, d2m, BWIDTH, color='r', label=lbl, bottom=d1m, zorder=3)
plt.title("CESCO (OAFA) sky conditions from MICA (1997-2012) and HASTA (2009-2020) data \n" +
          "(avg $\pm$ std excluding: M1997, M2012, H2009, H2020)")
#plt.yticks(np.arange(0, 110, 10))
plt.xticks(np.arange(1997, 2021, 1))
plt.legend(loc="lower right")
plt.grid(zorder=0)
plt.ylabel("hour / default obs. interval [%]")
plt.xlabel("year")
plt.savefig(OPATH+'/TABLE3_hours_year_default_interval'+IMGFMT, dpi=DPI)

# plots days per month
plt.subplots(1, 1, figsize=FIGSZ)
# hasta
y = np.arange(12)+1.5
d = np.array([float(h[i+3*NOYH][2]) for i in range(12)])
d1 = np.array([float(h[i+3*NOYH][4]) for i in range(12)])
d2 = np.array([float(h[i+3*NOYH][6]) for i in range(12)])
plt.bar(y, d, BWIDTH, color='g', zorder=3, alpha=0.6)
plt.bar(y, d2, BWIDTH, color='k', bottom=d, zorder=3, alpha=0.6)
plt.bar(y, d1, BWIDTH, color='r', bottom=d+d2, zorder=3, alpha=0.6)
# mica
y = np.arange(12)+1
dm = np.array([float(m[i+3*NOYM][2]) for i in range(12)])
d1m = np.array([float(m[i+3*NOYM][4]) for i in range(12)])
d2m = np.array([float(m[i+3*NOYM][6]) for i in range(12)])
lbl = "sunny (M); observed (H)=" + STRFMTS.format(np.mean(dm[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(dm[1:-1])) + " ; "+STRFMTS.format(np.mean(d[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d[1:-1]))+"%"
plt.bar(y, dm, BWIDTH, color='g', label=lbl, zorder=3)
lbl = "not observed(M;H)="+STRFMTS.format(np.mean(d2m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d2m[1:-1])) + " ; "+STRFMTS.format(np.mean(d2[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d2[1:-1]))+"%"
plt.bar(y, d2m, BWIDTH, color='k', label=lbl, bottom=dm, zorder=3)
lbl = "fully cloudy(M;H)=" + STRFMTS.format(np.mean(d1m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d1m[1:-1])) + " ; "+STRFMTS.format(np.mean(d1[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d1[1:-1]))+"%"
plt.bar(y, d1m, BWIDTH, color='r', label=lbl, bottom=dm+d2m, zorder=3)
plt.title("CESCO (OAFA) sky conditions from MICA (1997-2012, dark) and HASTA (2009-2020, light) data")
#plt.yticks(np.arange(0, 110, 10))
plt.xticks(np.arange(1, 13, 1))
plt.legend(loc="lower right")
plt.grid(zorder=0)
plt.ylabel("day / total days of the month [%]")
plt.xlabel("month")
plt.savefig(OPATH+'/TABLE4_days_month'+IMGFMT, dpi=DPI)

# plots hours per month / sunny obs interval
plt.subplots(1, 1, figsize=FIGSZ)
# hasta
y = np.arange(12)+1.5
d = np.array([float(h[i+3*NOYH+12][2]) for i in range(12)])
d1 = np.array([float(h[i+3*NOYH+12][4]) for i in range(12)])
d2 = np.array([float(h[i+3*NOYH+12][6]) for i in range(12)])
#plt.bar(y,d,BWIDTH, color='g', label="observed",zorder=3)
plt.bar(y, d1, BWIDTH, color='g', bottom=0, zorder=3, alpha=0.6)
plt.bar(y, d2, BWIDTH, color='r', bottom=d1, zorder=3, alpha=0.6)
# mica
y = np.arange(12)+1
dm = np.array([float(m[i+3*NOYM+12][2]) for i in range(12)])
d1m = np.array([float(m[i+3*NOYM+12][4]) for i in range(12)])+dm
d2m = np.array([float(m[i+3*NOYM+12][6]) for i in range(12)])
#plt.bar(y,dm,BWIDTH, color='g',zorder=3)
lbl = "good + moderate (M) ; good (H)=" + STRFMTS.format(np.mean(d1m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d1m[1:-1])) + " ; "+STRFMTS.format(np.mean(d1[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d1[1:-1]))+"%"
plt.bar(y, d1m, BWIDTH, color='g', label=lbl, bottom=0, zorder=3)
lbl = "bad (M) ; with clouds (H)="+STRFMTS.format(np.mean(d2m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d2m[1:-1])) + " ; "+STRFMTS.format(np.mean(d2[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d2[1:-1]))+"%"
plt.bar(y, d2m, BWIDTH, color='r', label=lbl, bottom=d1m, zorder=3)
plt.title("CESCO (OAFA) sky conditions from MICA (1997-2012, dark) and HASTA (2009-2020, light) data")
#plt.yticks(np.arange(0, 110, 10))
plt.xticks(np.arange(1, 13, 1))
plt.legend(loc="lower right")
plt.grid(zorder=0)
plt.ylabel("hour / sunny obs. interval  [%]")
plt.xlabel("month")
plt.savefig(OPATH+'/TABLE5_hours_month_sunny_interval'+IMGFMT, dpi=DPI)

# plots hours per month / default obs interval
plt.subplots(1, 1, figsize=FIGSZ)
# hasta
y = np.arange(12)+1.5
d = np.array([float(h[i+3*NOYH+24][2]) for i in range(12)])
d1 = np.array([float(h[i+3*NOYH+24][4]) for i in range(12)])
d2 = np.array([float(h[i+3*NOYH+24][6]) for i in range(12)])
#plt.bar(y,d,BWIDTH, color='g', label="observed",zorder=3)
plt.bar(y, d1, BWIDTH, color='g', bottom=0, zorder=3, alpha=0.6)
plt.bar(y, d2, BWIDTH, color='r', bottom=d1, zorder=3, alpha=0.6)
# mica
y = np.arange(12)+1
dm = np.array([float(m[i+3*NOYM+24][2]) for i in range(12)])
d1m = np.array([float(m[i+3*NOYM+24][4]) for i in range(12)])
d2m = np.array([float(m[i+3*NOYM+24][6]) for i in range(12)])
#plt.bar(y,dm,BWIDTH, color='g',zorder=3)
lbl = "sunny (M); good (H)=" + STRFMTS.format(np.mean(d1m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d1m[1:-1])) + " ; "+STRFMTS.format(np.mean(d1[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d1[1:-1]))+"%"
plt.bar(y, d1m, BWIDTH, color='g', label=lbl, bottom=0, zorder=3)
lbl = "cloudy (M); with clouds (H)="+STRFMTS.format(np.mean(d2m[1:-1]))+"$\pm$"+STRFMTS.format(
    np.std(d2m[1:-1])) + " ; "+STRFMTS.format(np.mean(d2[1:-1]))+"$\pm$"+STRFMTS.format(np.std(d2[1:-1]))+"%"
plt.bar(y, d2m, BWIDTH, color='r', label=lbl, bottom=d1m, zorder=3)
plt.title("CESCO (OAFA) sky conditions from MICA (1997-2012, dark) and HASTA (2009-2020, light) data")
#plt.yticks(np.arange(0, 110, 10))
plt.xticks(np.arange(1, 13, 1))
plt.legend(loc="lower right")
plt.grid(zorder=0)
plt.ylabel("hour / default obs. interval  [%]")
plt.xlabel("month")
plt.savefig(OPATH+'/TABLE6_hours_month_default_interval'+IMGFMT, dpi=DPI)
