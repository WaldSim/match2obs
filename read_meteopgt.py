import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use("bmh")

df_meteo = pd.read_csv("meteopgt1476.all", usecols=["Wind direction sector", "Wind speed class", "stabilty class"])
invalids1476 = np.load("invalids1476.npy")
df_meteo.drop(df_meteo.index[invalids1476], inplace=True)

df_meteo2 = pd.read_csv("meteopgt1008.all", usecols=["Wind direction sector", "Wind speed class", "stabilty class"])
invalids1008 = np.load("invalids1008.npy")
df_meteo2.drop(df_meteo2.index[invalids1008], inplace=True)

# etah einlesen
etah_1476 = np.load("etah_h_cat_1476.npy")
etah_1008 = np.load("etah_h_cat_1008.npy")

# etah histogramm plotten
binwidth = 1.0
plt.hist(etah_1476, bins=np.arange(min(etah_1476), max(etah_1476) + binwidth, binwidth))
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel(r"$\eta_{h}$", fontsize=15)
plt.show()

# Dataframe aus etah's machen und histogramme plotten
df = pd.DataFrame([])
for i in etah_1476:
    data = df_meteo.iloc[[i]]
    df = df.append(data)
plt.hist = df["Wind direction sector"].hist(bins=70)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind direction [°]", fontsize=15)
plt.show()

plt.hist = df["Wind speed class"].hist(bins=14)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind speed class [m/s]", fontsize=15)
plt.show()

plt.hist = df["stabilty class"].hist(bins=7)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("stability class [m/s]", fontsize=15)
plt.show()

# statistic über ausgewählte situationen:
direction = df["Wind direction sector"].value_counts()
speed = df["Wind speed class"].value_counts()
stab = df["stabilty class"].value_counts()

df2 = pd.DataFrame([])
for i in etah_1008:
    data2 = df_meteo2.iloc[[i]]
    df2 = df2.append(data2)
plt.hist = df2["Wind direction sector"].hist(bins=35)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind direction [°]", fontsize=15)
plt.show()

plt.hist = df2["Wind speed class"].hist(bins=9)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind speed class [m/s]", fontsize=15)
plt.show()

plt.hist = df2["stabilty class"].hist(bins=7)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("stability class [m/s]", fontsize=15)
plt.show()

# statistic über ausgewählte situationen:
direction2 = df2["Wind direction sector"].value_counts()
speed2 = df2["Wind speed class"].value_counts()
stab2 = df2["stabilty class"].value_counts()

wind_speed = df["Wind speed class"].to_numpy()
stab_class = df["stabilty class"].to_numpy()
wind_dir = df["Wind direction sector"].to_numpy()

# bsp.:
# weather_sit = df_meteo.iloc[[4]]

df_invalids_1008 = pd.DataFrame([])
df_meteo2_all = pd.read_csv("meteopgt1008.all", usecols=["Wind direction sector", "Wind speed class", "stabilty class"])
for i in invalids1008:
    data = df_meteo2_all.iloc[[i]]
    df_invalids_1008 = df_invalids_1008.append(data)

df_invalids_1008["Wind speed class"].plot.hist(bins=10)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind speed class [m/s]", fontsize=15)
plt.show()

df_invalids_1008["Wind direction sector"].plot.hist(bins=50)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind direction sector [m/s]", fontsize=15)
plt.show()

df_invalids_1008["stabilty class"].plot.hist(bins=20)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("stability class [m/s]", fontsize=15)
plt.show()

etah = df.index.value_counts()
etah2 = df2.index.value_counts()
ratio = len(etah) / 1424
ratio2 = len(etah2) / 951

df_invalids_1476 = pd.DataFrame([])
df_meteo_all = pd.read_csv("meteopgt1476.all", usecols=["Wind direction sector", "Wind speed class", "stabilty class"])
for i in invalids1476:
    data = df_meteo_all.iloc[[i]]
    df_invalids_1476 = df_invalids_1476.append(data)

df_invalids_1476["Wind speed class"].plot.hist(bins=10)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind speed class [m/s]", fontsize=15)
plt.show()

df_invalids_1476["Wind direction sector"].plot.hist(bins=50)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind direction sector [m/s]", fontsize=15)
plt.show()

df_invalids_1476["stabilty class"].plot.hist(bins=20)
#plt.title("", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("stability class [m/s]", fontsize=15)
plt.show()


# wind direction sector umrechnen in wind direction start:
df["Wind direction"] = df["Wind direction sector"] * 10 - 5
df.hist("Wind direction sector", bins=40, color="green", alpha=0.5)
df2["Wind direction"] = df2["Wind direction sector"] * 10 - 5


# wind direction sector
fig, ax = plt.subplots()

a_heights, a_bins = np.histogram(df['Wind direction sector'])
b_heights, b_bins = np.histogram(df2['Wind direction sector'], bins=a_bins)

width = (a_bins[1] - a_bins[0]) / 3

ax.bar(a_bins[:-1], a_heights, width=width, facecolor='purple')
ax.bar(b_bins[:-1] + width, b_heights, width=width, facecolor='g')
plt.show()

fig2, ax = plt.subplots()

ax.hist(df['Wind speed class'], bins=[0.3, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0, 6.75, 7.5], width=0.3,
        facecolor='purple', label="cat 1476", alpha=0.5)
ax.hist(df2['Wind speed class'], bins=[0.25, 0.75, 1.5, 2.5, 4.0, 5.5, 7.0], width=0.3, facecolor='g', label="cat 1008",
        alpha=0.5)
plt.legend()
plt.title("wind speed class of the chosen weather situations", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind speed class [m/s]", fontsize=15)
plt.savefig("plots/windspeedchoosen",bbox_inches='tight')
plt.show()

fig5, ax = plt.subplots()

ax.hist(df['Wind speed class'], bins=[0.3, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0, 6.75, 7.5], width=0.3,
        facecolor='purple', label="cat 1476", alpha=0.5)
# ax.hist(df2['Wind speed class'] , bins= [0.25, 0.75,1.5,2.5,4.0 , 5.5 , 7.0],width=0.3, facecolor='g',label ="cat 1008", alpha=0.5)
plt.legend()
plt.title("wind speed class of the chosen weather situations", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind speed class [m/s]", fontsize=15)
plt.savefig("plots/windspeedchoosenone",bbox_inches='tight')
plt.show()

fig3, ax = plt.subplots()

ax.hist(df['stabilty class'], bins=np.arange(1, 8, 1) - 0.15, width=0.3, facecolor='purple', label="cat 1476",
        alpha=0.5)
ax.hist(df2['stabilty class'], bins=np.arange(1, 8, 1) - 0.15, width=0.3, facecolor='g', label="cat 1008", alpha=0.5)
plt.legend()
plt.title("stability class of the chosen weather situations", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("stability class", fontsize=15)
plt.savefig("plots/stabclasschoosen",bbox_inches='tight')
plt.show()

fig6, ax = plt.subplots()

ax.hist(df['stabilty class'], bins=np.arange(1, 8, 1) - 0.15, width=0.3, facecolor='purple', label="cat 1476",
        alpha=0.5)
# ax.hist(df2['stabilty class'] , bins= np.arange(1,8,1)-0.15,width=0.3, facecolor='g',label ="cat 1008", alpha=0.5)
plt.legend()
plt.title("stability class of the chosen weather situations", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("stability class", fontsize=15)
plt.savefig("plots/stabclasschoosenone",bbox_inches='tight')
plt.show()

fig4, ax = plt.subplots()

ax.hist(df['Wind direction'], bins=np.arange(0, 360, 10) - 0.5, facecolor='purple', label="cat 1476", alpha=0.5)
ax.hist(df2['Wind direction'], bins=np.arange(0, 360, 10) - 0.5, facecolor='g', label="cat 1008", alpha=0.5)
plt.legend()
plt.title("wind direction class of the chosen weather situations", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind direction sector [°]", fontsize=15)
plt.savefig("plots/winddirectionchoosen",bbox_inches='tight')
plt.show()

fig7, ax = plt.subplots()

ax.hist(df['Wind direction'], bins=np.arange(0, 360, 10) - 0.5, facecolor='purple', label="cat 1476", alpha=0.5)
# ax.hist(df2['Wind direction'] , bins= np.arange(0,360, 10)-0.5, facecolor='g',label ="cat 1008", alpha=0.5)
plt.legend()
plt.title("wind direction class of the chosen weather situations", fontsize=18)
plt.ylabel("frequency", fontsize=15)
plt.xlabel("wind direction sector [°]", fontsize=15)
plt.savefig("plots/winddirectionchoosenone",bbox_inches='tight')

plt.show()
