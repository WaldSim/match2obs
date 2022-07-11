import pickle
import matplotlib.pyplot as plt
import numpy as np

from matchobs_functions import matchgralorgramm_jul
from math_functions import arctan, windrose, filter_nans
from readobs import load_observed_jul
from settings import START2, END2, STATION_NAMES_JUL, GRAL, LOAD_CATALOGUE, N_CATALOGUE_USED

# load catalog
catalogue= matchgralorgramm_jul(GRAL, LOAD_CATALOGUE, N_CATALOGUE_USED) #alternativ
# direction and horizontal wind speed of catalog
cat_dir = arctan(catalogue[1, :, :], catalogue[0, :, :])
cat_hor = np.sqrt(np.sum(catalogue[0:1, :, :].astype(np.float32) ** 2, axis=0))

# plot windroses for 14 stations
for k in range(0, 14):
    windrose(cat_dir[k, :], cat_hor[k, :], f"catalog: {STATION_NAMES_JUL[k]}")
    plt.savefig(f"plots/articificalwindrose{STATION_NAMES_JUL[k]}")
    plt.show()
    plt.close()

# load observations and plot windroses for 14 stations
observations = load_observed_jul(START2, END2)
obs = filter_nans(observations[:, 1:3, :].astype(np.float32))
for k in range(0, 14):
    windrose(obs[:, 1, k], obs[:, 0, k], f"observation: {STATION_NAMES_JUL[k]}")
    plt.savefig(f"plots/winroseobserved{STATION_NAMES_JUL[k]}")
    plt.show()
    plt.close()
