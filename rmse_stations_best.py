import itertools
import pickle

import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt

from matchobs_functions import calculate_rmse, matchedwinds, choose_metric, min_score
from readobs import load_observed_jul
from settings import METRIK, START2, END2, USE_WEIGHTING, STATION_COLORS


def match_with_stations(observation, catalogue, stat_list,
                        use_weighting,
                        weight=1.,
                        use_smoothing=False,
                        metric="l2"):
    scores = choose_metric(observation, catalogue, stat_list, metric,
                           use_weighting)
    eta_h = min_score(scores)
    return eta_h


if __name__ == "__main__":

    observations = load_observed_jul(START2, END2)
    # def used_stations_plot(observations, catalogue_winds_uv, USE_WEIGHTING, cat_idx, nstations, keys):
    keys = {"rmse_all",
            # "rmse_stations"
            }
    # N_STATIONS = 4
    N_STATIONS = 14
    cat_idx = 1476
    pkl_file = open('winds_cat1476_final.pkl', 'rb')
    catalogue = pickle.load(pkl_file)
    pkl_file.close()
    N_CATALOGUE = 1424
    observations = observations[:, :, :N_STATIONS + 1]
    catalogue = catalogue[:, :N_STATIONS + 1, :]

    # observations=observations[:, :,N_STATIONS]
    # catalogue = catalogue[:, :N_STATIONS, :]
    full_results = {
        key: {
            "min": list(),
            "index_min": list(),
            "max": list(),
            "index_max": list(),
            "mean": list(),
            "combis": list()
        } for key in keys
    }
    all_stations = [n for n in range(N_STATIONS)]
    for i in range(1, N_STATIONS + 1):
        rmses = [list() for _ in range(len(keys))]
        station_combinations = itertools.combinations(all_stations, i)
        combi = list(itertools.combinations(all_stations, i))

        for station_list in station_combinations:
            print(f"matching stations {station_list}")
            etah = match_with_stations(observations, catalogue, station_list, USE_WEIGHTING)
            matched_winds = matchedwinds(catalogue, etah)
            matched_winds2 = rearrange(matched_winds, "d s h  -> h d s ")
            results_rmse = calculate_rmse(observations[:, 3:5, :], matched_winds2[:, :, :])

            for i, key in enumerate(keys):
                rmses[i].append(results_rmse[key])

        for i in range(len(rmses)):
            rmses[i] = np.array(rmses[i])

        for key, rmse in zip(keys, rmses):
            full_results[key]["min"].append(rmse.min())
            full_results[key]["index_min"].append((rmse.argmin()))
            full_results[key]["max"].append(rmse.max())
            full_results[key]["index_max"].append((rmse.argmax())),
            full_results[key]["mean"].append(rmse.mean())
            full_results[key]["combis"].append(combi)

    min = full_results[key]["min"]
    min_index = full_results[key]["index_min"]
    max = full_results[key]["max"]
    max_index = full_results[key]["index_max"]
    mean = full_results[key]["mean"]
    combis = full_results[key]["combis"]
    np.save(f'arrays/min_all{N_STATIONS}.npy', min)
    np.save(f'arrays/max_all{N_STATIONS}.npy', max)
    np.save(f'arrays/min_index_all{N_STATIONS}.npy', min_index)
    np.save(f'arrays/max_index_all{N_STATIONS}.npy', max_index)
    np.save(f'arrays/mean_all{N_STATIONS}.npy', mean)
    np.save(f'arrays/combis_all{N_STATIONS}.npy', combis)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, N_STATIONS + 1), full_results[key]["min"], lw=np.pi, color="black",
             label=r"$\overline{RMSE}_{min}$")
    ax2.plot(np.arange(1, N_STATIONS + 1), full_results[key]["max"], lw=np.pi, color="blue",
             label=r"$\overline{RMSE_{max}}$")
    ax2.plot(np.arange(1, N_STATIONS + 1), full_results[key]["mean"], lw=np.pi, color="purple",
             label=r"$\overline{RMSE_{max}}$")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax1.set_ylabel(r"$\Delta RMSE$", fontsize=18)
    ax1.set_xlabel("number of stations included", fontsize=20)
    ax2.set_ylim(0, 3.0)
    ax2.set_ylabel(r"$\overline{RMSE}$")
    plt.xlim(1, 14)
    ax1.legend()
    ax2.legend()
    plt.title("RMSE vs. number of included stations")
    fig.tight_layout()
    plt.savefig(f"plots/RMSEvsstations_gral_jul_okt{N_STATIONS}_min_max_mean.png", bbox_inches='tight')
    plt.show()
    plt.close()

    print("done done.")
