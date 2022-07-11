import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
import settings
from matchobs_functions import matchgralorgramm_jul, obshor
from math_functions import arctan
from readobs import load_observed_jul

if __name__ == '__main__':
    # print all the settings
    print(f"used_stations:{settings.USED_STATIONS}")
    print(f"stations:{settings.STATION_NAMES_JUL}")
    print(f"start:{settings.START2}")
    print(f"end:{settings.END2}")
    print(f"metrik:{settings.METRIK}")
    print(f"n_catalog:{settings.N_CATALOGUE_USED}")
    print(f"smoothing:{settings.USE_SMOOTHING}")
    print(f"load_cat:{settings.LOAD_CATALOGUE}")
    print(f"gral:{settings.GRAL}")
    print(f"use_weighting:{settings.USE_WEIGHTING}")
    # load catalogue
    catalogue = matchgralorgramm_jul(settings.GRAL, settings.LOAD_CATALOGUE, settings.N_CATALOGUE_USED)
    #print(np.shape(catalogue))
    # load observations
    observations = load_observed_jul(settings.START2, settings.END2)
    obs_hor = obshor(observations)
    cat_dir = arctan(catalogue[1, :, :].astype(np.float32), catalogue[0, :, :].astype(np.float32))
    obs_dir = observations[:, 2, :, ]
    diffwinkel = (observations[:, 2, :, None] - cat_dir[None, :, :]).astype(np.float32)
    diffwinkel2 = (diffwinkel + 180) % 360 - 180
    a = (diffwinkel2 * np.pi) / 180
    diff1 = np.cos(a)
    diff2 = np.sqrt((np.sqrt(np.sum(observations[:, 3:5, :, None].astype(float) ** 2, axis=1))
                     - np.sqrt(np.sum(catalogue[None, :-1, :, :].astype(float) ** 2, axis=1))) ** 2)
    diff_square = diff2
    obs_min = np.nanmedian(observations[:, 1, :, ].astype(float))
    obs_windspeed = observations[:, 1, :, ]
    obs_windspeed[obs_windspeed <= obs_min] = obs_min
    obs_windspeed_mean = (np.nanmean(obs_windspeed[:, :].astype(float), axis=0))
    obs_windspeedmean = obs_windspeed_mean[None, :, None]
    rmse_hor = []
    rmse_dir = []
    rmse = []
    mean_bias_hor = []
    mean_bias_dir = []
    lam = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.48, 0.46, 0.44, 0.42, 0.4, 0.3, 0.2, 0.1, 0]
    for l in (lam):
        eta_matrix = np.nansum(l * (diff_square / obs_windspeedmean) - (1 - l) * diff1,
                               axis=1)
        eta_matrix[eta_matrix == 0] = np.nan
        eta_h = np.argmin(eta_matrix, axis=1)
        matched_winds = catalogue[:-1, :, eta_h]
        matched_winds2 = rearrange(matched_winds, "d s h  -> h d s ")
        matched_hor = np.sqrt(
            np.sum(matched_winds2.astype(np.float32) ** 2, axis=1))  # horizontal wind speed of matched winds
        match_hor = np.sqrt(np.sum(matched_winds2.astype(np.float32) ** 2, axis=1))
        match_dir = arctan(matched_winds2[:, 1, :], matched_winds2[:, 0, :])
        a = obs_dir - match_dir
        winkel_diff = (a + 180) % 360 - 180
        # calculate rmse, mb
        rmse_all = np.sqrt(np.nanmean((np.sum((observations[:, 3:5, :].astype(float) - matched_winds2) ** 2, axis=1))))
        rmse_all_hor = np.sqrt(np.nanmean(((obs_hor.astype(np.float32) - match_hor.astype(np.float32)) ** 2)))
        rmse_all_dir = np.sqrt(np.nanmean(winkel_diff.astype(np.float32) ** 2))
        mbe_hor = np.nanmean(obs_hor - matched_hor)
        mbe_all_dir = np.nanmean((winkel_diff.astype(np.float32)))
        mean_bias_hor.append(mbe_hor)
        mean_bias_dir.append(mbe_all_dir)
        rmse_dir.append(rmse_all_dir)
        rmse_hor.append(rmse_all_hor)
        rmse.append(rmse_all)

    # plt.plot(lam,mean_bias_hor)
    # plt.show()
    fig = plt.figure()
    plt.plot(lam, rmse_hor, label="RMSE wind speed [m/s]")
    plt.title(r"RMSE of wind speed with modified $\lambda$", fontsize=18)
    plt.legend()
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel("RMSE of wind speed [m/s]", fontsize=15)
    plt.savefig(f"plots/anisotropicrmsewindspeed.png", bbox_inches='tight')
    plt.show()

    # plt.plot(lam,mean_bias_dir)
    # plt.show()
    fig2 = plt.figure()
    plt.plot(lam, rmse_dir, label="RMSE wind direction [°]")
    plt.title(r"RMSE of wind direction with modified $\lambda$", fontsize=18)
    plt.legend()
    plt.xlabel(r"$\lambda$", fontsize=15)
    plt.ylabel("RMSE of wind direction [°]", fontsize=15)
    plt.savefig(f"plots/anisotropicrmsewinddirection.png", bbox_inches='tight')
    plt.show()
    # plt.plot(lam, rmse)
    # plt.show()

    fig3, ax1 = plt.subplots()
    ax1.plot(lam, rmse_hor, label="RMSE wind speed [m/s]", color="green")
    ax1.plot(lam, rmse, label="RMSE(u,v) [m/s]", color="black")
    ax2 = ax1.twinx()
    ax2.plot(lam, rmse_dir, label="RMSE wind direction [°]", color="purple")
    ax1.set_xlabel(r"$\lambda$", fontsize=20)
    ax1.set_ylabel("RMSE [m/s]", fontsize=15)
    ax2.set_ylabel("RMSE of wind direction [°]", fontsize=15)
    plt.title(r"RMSE when varying lambda $\lambda$", fontsize=18)
    ax1.legend(loc=0)
    ax2.legend(loc=1)
    plt.savefig(f"plots/anisotropicmetriclambda.png", bbox_inches='tight')
    plt.show()
