import numpy as np
import matplotlib.pyplot as plt
import itertools
from settings import USE_WEIGHTING, METRIK, N_CATALOGUE_USED, USED_STATIONS, START2, END2, GRAL, LOAD_CATALOGUE
from einops import rearrange
from matchobs_functions import calculate_weight, min_score, choose_metric, matchedwinds, calculate_rmse, \
    matchedwinds_rmse, matchgralorgramm_jul
from readobs import load_observed_jul

N_STATIONS = len(USED_STATIONS)


def match_with_stations(observation, catalogue, stat_list,
                        use_weighting,
                        weight=1.,
                        use_smoothing=False,
                        metric="l2vobsminmean"):
    scores = choose_metric(observation, catalogue, stat_list, metric,
                           use_weighting)
    eta_h = min_score(scores)
    return eta_h


min_num_stations = N_STATIONS

def match_and_rmse(obsuv, cat, station_list, cat_idx, use_weighting, verbose=False):
    catuv = cat[:-1, station_list, cat_idx]
    etah = match_with_stations(obsuv, cat, station_list, use_weighting)

    matched_winds = matchedwinds_rmse(catuv, etah, station_list)
    matched_winds2 = rearrange(matched_winds, "d s h  -> h d s ")
    results_rmse = calculate_rmse(observations[:, 3:5, :], matched_winds2[:, :, :])
    return results_rmse


def used_stations_plot(observations, catalogue_winds_uv, USE_WEIGHTING, cat_idx, nstations, keys):
    full_results = {
        key: {
            "mean": list()
        } for key in keys
    }
    all_stations = [n for n in range(nstations)]
    for i in range(nstations + 1):
        rmses = [list() for _ in range(len(keys))]
        station_combinations = itertools.combinations(all_stations, i)
        for station_list in station_combinations:
            print(f"matching stations {station_list}")
            # weight = calculate_weight(observations, station_list, USE_WEIGHTING)
            results = match_and_rmse(observations,
                                     catalogue_winds_uv,
                                     station_list,
                                     cat_idx - 1,
                                     USE_WEIGHTING,
                                     verbose=True,
                                     )
            for i, key in enumerate(keys):
                rmses[i].append(results[key])

        for i in range(len(rmses)):
            rmses[i] = np.array(rmses[i])

        for key, rmse in zip(keys, rmses):
            full_results[key]["mean"].append(rmse.mean())

    return full_results


def plot_and_calcaulate_rmse(observations, catalogue_winds_uv, cat_idx,
                             use_weighting, savename="rmses.png",
                             keys=["rmse_all"],
                             min_num_stations=1, plot=True):
    observations_uv = observations[:, 3:5, :].astype(np.float32)  # 168 (time), 2 (u,v), 8 (n_stations)
    # plotting and storing results
    full_results = {
        key: {
            "mean": list(),
            "std": list()
        } for key in keys
    }
    # all possible combinations
    all_stations = [n for n in range(N_STATIONS)]
    for i in range(min_num_stations, N_STATIONS + 1):  # how many stations to include for match2obs
        rmses = [list() for _ in range(len(keys))]
        station_combinations = itertools.combinations(all_stations, i)
        for station_list in station_combinations:
            print(f"matching stations {station_list}")
            weight = calculate_weight(observations, station_list, USE_WEIGHTING)
            results = match_and_rmse(observations,
                                     catalogue_winds_uv,
                                     station_list,
                                     cat_idx,
                                     use_weighting,
                                     verbose=True,
                                     )

            for i, key in enumerate(keys):
                rmses[i].append(results[key])

        for i in range(len(rmses)):
            rmses[i] = np.array(rmses[i])

        for key, rmse in zip(keys, rmses):
            full_results[key]["mean"].append(rmse.mean())
            full_results[key]["std"].append(rmse.std())

    if plot:
        for key in keys:
            # plt.errorbar(np.arange(1, N_STATIONS + 1), full_results[key]["mean"],
            #             yerr=full_results[key]["std"],
            #             label=key)

            plt.plot(np.arange(1, N_STATIONS + 1), full_results[key]["mean"], lw=np.pi, color="darkblue")
            plt.plot(np.arange(1, N_STATIONS + 1),
                     np.array(full_results[key]["mean"]) + np.array(full_results[key]["std"]),
                     color="darkblue", lw=np.pi / 2)
            plt.fill_between(np.arange(1, N_STATIONS + 1), full_results[key]["mean"],
                             np.array(full_results[key]["mean"]) + np.array(full_results[key]["std"]),
                             alpha=0.2, color="darkblue")
            plt.plot(np.arange(1, N_STATIONS + 1),
                     np.array(full_results[key]["mean"]) - np.array(full_results[key]["std"]),
                     color="darkblue", lw=np.pi / 2)
            plt.fill_between(np.arange(1, N_STATIONS + 1), np.array(full_results[key]["mean"]),
                             np.array(full_results[key]["mean"]) - np.array(full_results[key]["std"]), alpha=0.2,
                             color="darkblue")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel("RMSE", fontsize=20)
        plt.xlabel("number of stations included", fontsize=20)
        # plt.legend()
        # plt.title(f"RMSE at 8 stations: {full_results['rmse_full']['mean'][-1]:.2f}, {n_cat} catalogue entries", fontsize=18)
        plt.title("RMSE vsnumber of included stations (for 1008 cat. entries)")
        plt.savefig(savename)
        plt.close()
    return full_results


if __name__ == "__main__":
    # load catalog
    catalogue_winds = matchgralorgramm_jul(GRAL, LOAD_CATALOGUE, N_CATALOGUE_USED)
    catalogue_winds_uv = catalogue_winds[:-1, ...]  # 2 (u,v), n_stations, n_catalogue
    # with open("catalogue_winds.npy", "wb") as f:
    #    np.save(f, catalogue_winds)
    # load observations:
    observations = load_observed_jul(START2, END2)
    # with open("observations.npy", "wb") as f:
    # np.save("observations.npy", observations)
    # extract wind vectors (u, v , i.e. horizontal components)
    observations_uv = observations[:, 3:5, :].astype(np.float32)

    # check for nans and count them:
    print(f"{np.count_nonzero(~np.isnan(observations_uv))} non-nans, "
          f"{np.count_nonzero(np.isnan(observations_uv))} nans "
          f"{np.prod(observations_uv.shape)} total entries in observations_uv.")

    # vary number of used CAT entries
    n_cat_list = ([2 ** i for i in range(10)] + [1424])
    # n_cat_list = [1008]

    keys = [
        "rmse_all",
        # "rmse_u", "rmse_v"
    ]

    n_stat = 30
    last_values = {
        key: [list() for _ in range(n_stat)] for key in keys
    }

    full_results = {
        key: {
            "mean": list(),
            "std": list()
        } for key in keys
    }
    #  rmse vs n_catalogue
    for n_cat in n_cat_list:
        print(f"matching for n_cat = {n_cat}...")
        n_cat_results = {key: np.zeros((n_stat,)) for key in keys}
        for i in range(n_stat):  # simulate some stats
            cat_idx = np.random.choice(np.arange(N_CATALOGUE_USED - 52), size=(n_cat),
                                       replace=False)  # -52 da nicht alle wettersituationen verwendet werden
            # cat_idx = np.arange(n_cat)
            catalogue = catalogue_winds[:, :, cat_idx]  # 2 (u,v), n_stations, n_catalogue
            # all possible combinations
            station_list = [n for n in range(N_STATIONS)]
            print(f"matching stations {station_list}")
            # weight = calculate_weight(observations, station_list, USE_WEIGHTING)
            etah = match_with_stations(observations, catalogue, USED_STATIONS, USE_WEIGHTING, METRIK)
            matched_winds = matchedwinds(catalogue, etah)
            matched_winds = rearrange(matched_winds, "d s h  -> h d s ")
            results = calculate_rmse(observations[:, 3:5, :], matched_winds[:, :, :])

            for key in keys:
                n_cat_results[key][i] = results[key]

        for k, key in enumerate(keys):
            full_results[key]["mean"].append(n_cat_results[key].mean())
            full_results[key]["std"].append(n_cat_results[key].std())

    for key in full_results.keys():
        plt.figure(figsize=(9, 6))
        m = np.array(full_results[key]["mean"])  # n_cat
        s = np.array(full_results[key]["std"])  # n_cat
        # plt.errorbar(np.arange(len(n_cat_list)), x.mean(0), yerr=x.std(0)/np.sqrt(n_stat), label=key)
        plt.plot(np.arange(len(n_cat_list)), m, lw=np.pi, color="darkblue", label="RMSE")
        plt.plot(np.arange(len(n_cat_list)), m + s / np.sqrt(n_stat), color="darkblue", lw=np.pi / 2,
                 label=r"$\sigma_{RMSE}$")
        plt.fill_between(np.arange(len(n_cat_list)), m, m + s / np.sqrt(n_stat), alpha=0.2, color="darkblue")
        plt.plot(np.arange(len(n_cat_list)), m - s / np.sqrt(n_stat), color="darkblue", lw=np.pi / 2)
        plt.fill_between(np.arange(len(n_cat_list)), m, m - s / np.sqrt(n_stat), alpha=0.2, color="darkblue")
        plt.xlabel("number of used catalog entries", fontsize=18)
        plt.xticks(np.arange(len(n_cat_list)), n_cat_list, fontsize=15)
        plt.yticks(fontsize=18)
        plt.legend()
        plt.ylabel("RMSE [m/s]", fontsize=18)
        plt.title("RMSE vs. number of used catalog entries", fontsize=20)
        plt.savefig(f"plots/{key}_rmse_vs_catalogue_1476.png",bbox_inches='tight')
        plt.show()
        plt.close()

    print("done.")
