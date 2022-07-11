import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use("bmh")
df_meteo = pd.read_csv("messdaten/utc/theo_global_all_jul_okt_21_utc.csv")

df_meteo.hist("solarRadiation")
plt.savefig("plots/solar_radiation.png")
plt.show()

