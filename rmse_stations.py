import itertools
import pickle
import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from matchobs_functions import calculate_rmse, matchedwinds, choose_metric, min_score, matchgralorgramm_jul
from readobs import load_observed_jul
from settings import START2, END2, USE_WEIGHTING, GRAL, LOAD_CATALOGUE, N_CATALOGUE_USED, METRIK, STATION_NAMES_JUL


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
    catalogue = matchgralorgramm_jul(GRAL, LOAD_CATALOGUE, N_CATALOGUE_USED)

    # def used_stations_plot(observations, catalogue_winds_uv, USE_WEIGHTING, cat_idx, nstations, keys):
    keys = {"rmse_all",
            # "rmse_stations"
            }
    # N_STATIONS = 4
    N_STATIONS = 14
    cat_idx = 1476
    observations = observations[:, :, :N_STATIONS + 1]
    catalogue = catalogue[:, :N_STATIONS + 1, :]

    # observations=observations[:, :,N_STATIONS]
    # catalogue = catalogue[:, :N_STATIONS, :]
    full_results = {
        key: {
            "mean": list()
        } for key in keys
    }
    all_stations = [n for n in range(N_STATIONS)]
    for i in range(1, N_STATIONS + 1):
        rmses = [list() for _ in range(len(keys))]
        station_combinations = itertools.combinations(all_stations, i)
        for station_list in station_combinations:
            print(f"matching stations {station_list}")
            etah = match_with_stations(observations, catalogue, station_list, USE_WEIGHTING)
            matched_winds = matchedwinds(catalogue, etah)
            # matched_winds = matchedwinds(catalogue[:, :, :], etah)
            matched_winds2 = rearrange(matched_winds, "d s h  -> h d s ")
            # only used station for rmse calculation:
            results_rmse = calculate_rmse(observations[:, 3:5, station_list], matched_winds2[:, :, station_list])
            # results_rmse = calculate_rmse(observations[:, 3:5, :], matched_winds2[:, :, :])

            for i, key in enumerate(keys):
                rmses[i].append(results_rmse[key])

        for i in range(len(rmses)):
            rmses[i] = np.array(rmses[i])

        for key, rmse in zip(keys, rmses):
            full_results[key]["mean"].append(rmse.mean())

    # Verwende für alle Stationen

    # for every station all combinations
    keys = {  # "rmse_all",
        "rmse_station"
    }

    full_results_stations = list()
    rmse_results_stations = []
    rmse_all_stations = []
    rmse_all = []
    rmse_mean_all = []
    rmse_mean_all_is = []
    # all_s = [0, 1, 2, 3]
    all_s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    for station in all_s:
        others = all_s.copy()
        others.remove(station)

        rmse_stations = []
        rmse_mean_all_i = []
        for i in range(0, len(others) + 1):

            other_combinations = itertools.combinations(others, i)
            # print(other_combinations)
            for other_stations in other_combinations:
                station_list = [station] + list(other_stations)
                rmse_stations.append(station_list)
                etah = match_with_stations(observations, catalogue, station_list, USE_WEIGHTING, METRIK)
                matched_winds = matchedwinds(catalogue, etah)
                matched_winds2 = rearrange(matched_winds, "d s h  -> h d s ")
                results_rmse = calculate_rmse(observations[:, 3:5, :], matched_winds2[:, :, :])
                rmse_results_stations.append(results_rmse["rmse_all"])
                print(f"{station_list}")
            # print(f"{rmse_stations}")
            # rmse_all_stations.append(rmse_stations)
            rmse_mean_all_i.append(results_rmse["rmse_all"].mean())
        rmse_mean_all_is.append(rmse_mean_all_i)
        # rmse_mean_all.append(results_rmse["rmse_all"])
        # rmse_all.append(rmse_all_stations)

    rmse_mean_all_is = np.asarray(rmse_mean_all_is)
    rmse_diff = full_results[key]["mean"][-1] - rmse_mean_all_is
    np.save(f"arrays/rmse_diff_{N_STATIONS}_1476.npy", rmse_diff)
    mean = full_results[key]["mean"]
    np.save(f'arrays/mean_{N_STATIONS}_1476.npy', mean)
    STATION_COLORS = {0: "maroon", 1: "darkorange", 2: 'gold', 3: 'darkolivegreen', 4: 'turquoise', 5: 'slategrey',
                      6: 'darkcyan', 7: 'magenta',
                      8: "blue", 9: "black", 10: "grey", 11: "yellow", 12: "green", 13: "pink"}

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, N_STATIONS + 1), full_results[key]["mean"], lw=np.pi, color="black",
             label=r"$\overline{RMSE}$")
    # plt.plot(np.arange(1, N_STATIONS + 1), np.array(full_results[key]["mean"]) + np.array(full_results[key]["std"]),
    #         color="darkblue", lw=np.pi / 2)
    # plt.fill_between(np.arange(1, N_STATIONS + 1), full_results[key]["mean"],
    #                 np.array(full_results[key]["mean"]) + np.array(full_results[key]["std"]),
    #                 alpha=0.2, color="darkblue")
    # plt.plot(np.arange(1, N_STATIONS + 1), np.array(full_results[key]["mean"]) - np.array(full_results[key]["std"]),
    #         color="darkblue", lw=np.pi / 2)
    # plt.fill_between(np.arange(1, N_STATIONS + 1), np.array(full_results[key]["mean"]),
    #                 np.array(full_results[key]["mean"]) - np.array(full_results[key]["std"]), alpha=0.2,
    #                 color="darkblue")
    # differenzen:
    for k in range(0, N_STATIONS):
        ax1.plot(np.arange(1, N_STATIONS + 1), rmse_diff[k], color=f"{STATION_COLORS[k]}",
                 label=f"{STATION_NAMES_JUL[k]}")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax1.set_ylabel(r"$\Delta RMSE [m/s]$", fontsize=18)
    ax1.set_xlabel("number of stations included", fontsize=20)
    ax2.set_ylim(0, 3.0)
    ax2.set_ylabel(r"$\overline{RMSE} [m/s]$")
    plt.xlim(1, 14)
    ax1.legend()
    ax2.legend()
    plt.title("RMSE vs. number of included stations")
    fig.tight_layout()
    plt.savefig(f"plots/RMSEvsstations_gral_jul_okt{N_STATIONS}_1476.png", bbox_inches='tight')
    plt.show()
    plt.close()

    print("done.")
