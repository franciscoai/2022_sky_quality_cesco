import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

REPO_PATH = os.getcwd()
path = REPO_PATH + '/data/master'
OPATH = REPO_PATH + '/output/master'  

# create opath
os.makedirs(OPATH, exist_ok=True)

# read all files in path
file = pd.read_fwf(path)
file["DAY"] = pd.to_datetime(file["DAY"]+'T'+file["Hour"], format='%Y%m%dT%H:%M:%S.s')

#plot all columns using a for
x = file["DAY"]
y = file["Sky_T"]
plt.scatter(x, y, c="red")
plt.xlabel("Day")
plt.ylabel("Sky_T")
plt.title("Sky_T/DAY")
plt.show()


x = file["DAY"]
y = file["Amb_T"]
plt.scatter(x, y, c="blue")
plt.xlabel("Day")
plt.ylabel("Amb_T")
plt.title("Amb_T/DAY")
plt.show()


x = file["DAY"]
y = file["WindS"]
plt.scatter(x, y, c="green")
plt.xlabel("Day")
plt.ylabel("Winds")
plt.title("Winds/DAY")
plt.show()

x = file["DAY"]
y = file["Hum%"]
plt.scatter(x, y, c="purple")
plt.xlabel("Day")
plt.ylabel("Hum%")
plt.title("Hum%/DAY")
plt.show()


x = file["DAY"]
y = file["DewP"]
plt.scatter(x, y, c="black")
plt.xlabel("Day")
plt.ylabel("DewP")
plt.title("DewP/DAY")
plt.show()


x = file["DAY"]
y = file["C"]
plt.scatter(x, y, c="pink")
plt.xlabel("Day")
plt.ylabel("C")
plt.title("C/DAY")
plt.show()


x = file["DAY"]
y = file["W"]
plt.scatter(x, y, c="orange")
plt.xlabel("Day")
plt.ylabel("W")
plt.title("W/DAY")
plt.show()


x = file["DAY"]
y = file["R"]
plt.scatter(x, y, c="yellow")
plt.xlabel("Day")
plt.ylabel("R")
plt.title("R/DAY")
plt.show()

Cloud_T = file["Sky_T"]-file["Amb_T"]

x = file["DAY"]
y = Cloud_T
plt.scatter(x, y, c="grey")
plt.xlabel("Day")
plt.ylabel("Cloud_T")
plt.title("Cloud_T/DAY")
plt.show()
