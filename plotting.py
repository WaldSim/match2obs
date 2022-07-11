import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from scipy.stats import stats

plt.style.use('bmh')


def plot_matching_stations(observations, matchedwinds, station_names, metric, use_weight, use_smooth, ncols, nrows):
    ncols = ncols
    nrows = nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 15), sharex=True, sharey=True)
    # fig.suptitle(f"Metrik: {metric}, RMSE_hor: {rmse_all_stat:.2f}, Mean bias_hor: {mean_bias_all:.2f} ", fontsize=26)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            try:
                axes[i, j].grid(True, linewidth=0.5, color="black", linestyle="-")
                # axes[i, j].plot(obs_hor_smooth[k, :], label="wind observed_smooth", marker="", color="darkblue", linestyle="--")
                axes[i, j].plot(observations[:, k], label="wind observed", marker="", color="darkblue", linestyle="--")
                axes[i, j].plot(matchedwinds[:, k], label=f"wind simulated {station_names[k]}", marker="",
                                linestyle="-",
                                color="orange", alpha=0.8)
                axes[i, j].set_title(f"{station_names[k]}")
                axes[i, j].legend(loc="upper right")
                axes[i, j].set_title(f"matched and observed wind for measuring station {station_names[k]}", fontsize=24)
                fig.text(0.5, 0.07, 'time [h]', ha='center', fontsize=30)
                fig.text(0.07, 0.5, 'wind speed [m/s]', va='center', rotation='vertical', fontsize=30)
                k += 1
                plt.subplots_adjust(left=0.11,
                                    bottom=0.15,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.08,
                                    hspace=0.35)
            except IndexError:
                axes[i, j].axis("off")
    plt.savefig(f"plots/windspeed{metric}_weight_{use_weight}_smooth_{use_smooth}_hor_GRAMM_jul_okt.png", bbox_inches='tight')
    plt.figure()
    plt.show()

def plot_matching_stations_cats(observations, matchedwinds,matchedwinds2, station_names , metric, use_weight, use_smooth, ncols, nrows):
    ncols = ncols
    nrows = nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 15), sharex=True, sharey=True)
    #fig.delaxes(axes[3, 1])
    #fig.suptitle(f"Metrik: {metric}, RMSE_hor: {rmse_all_stat:.2f}, Mean bias_hor: {mean_bias_all:.2f} ", fontsize=26)

    k = 0
    for i in range(nrows):

        for j in range(ncols):
            try:
                #for k in ~[used_stations]:
                 #   plt.rcParams.update("grey")
                axes[i, j].grid(True, linewidth=0.5, color="black", linestyle="-")

                # axes[i, j].plot(obs_hor_smooth[k, :], label="wind observed_smooth", marker="", color="darkblue", linestyle="--")
                axes[i, j].plot(observations[:, k], label="wind observed", marker="", color="darkblue", linestyle="--")

                axes[i, j].plot(matchedwinds[:, k], label=f"wind simulated {station_names[k]}", marker="",linestyle="-",
                                color="orange", alpha=0.8)

                axes[i, j].plot(matchedwinds2[:, k], label=f"wind simulated 2 {station_names[k]}", marker="",linestyle="-",
                                color="green", alpha=0.8)

                axes[i, j].set_title(f"{station_names[k]}")
                axes[i, j].legend(loc="upper right")

                axes[i, j].set_title(f"matched and observed wind for measuring station {station_names[k]}", fontsize=24)
                fig.text(0.5, 0.07, 'time [h]', ha='center', fontsize=30)
                fig.text(0.07, 0.5, 'wind speed [m/s]', va='center', rotation='vertical', fontsize=30)
                k += 1
                plt.subplots_adjust(left=0.11,
                                    bottom=0.15,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.08,
                                    hspace=0.35)
            except IndexError:
                axes[i,j].axis("off")
            plt.savefig(f"plots/windspeed{metric}_weight_{use_weight}_smooth_{use_smooth}_hor_GRAMM_jul_okt.png", bbox_inches='tight')
    plt.figure()
    plt.show()


def lin_reg(x, y, color="blue", ax=None, alpha=1., ):
    xisnanindex = np.isnan(x)
    x_data = x[~xisnanindex]
    y_data = y[~xisnanindex]
    res = stats.linregress(x_data, y_data)
    print((f"R-squared: {res.rvalue ** 2:.6f}"))
    # plot the data:
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.yaxis.grid(color="gray", linestyle="dashed")
    plt.plot(x, y, 'o', label='original data', color=color, alpha=alpha)
    plt.plot(x, res.intercept + res.slope * x, 'r', label='fitted line', color=color)
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
    at = AnchoredText(f"R-squared: {res.rvalue ** 2:.6f}",
                      prop=dict(size=15), frameon=True,
                      loc='upper right',
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    plt.ylabel("matched wind speed m/s")
    plt.xlabel("observed wind speed m/s")
    plt.legend()
    # plt.show()


def lin_reg_stations(x, y, color="blue", ax=None, alpha=1., label="fitted line", label_orig="original data"):
    xisnanindex = np.isnan(x)
    x_data = x[~xisnanindex]
    y_data = y[~xisnanindex]
    res = stats.linregress(x_data, y_data)
    print((f"R-squared: {res.rvalue ** 2:.6f}"))
    # plot the data:
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.yaxis.grid(color="gray", alpha=0.5)
        ax.plot(x, y, 'o', label=label_orig, color=color, alpha=alpha)
        ax.plt.plot(x, res.intercept + res.slope * x, 'r', label=label, color=color)
        xpoints = ypoints = plt.xlim()
        ax.plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
        at = AnchoredText(f"R-squared: {res.rvalue ** 2:.2f}",
                          prop=dict(size=15), frameon=True,
                          loc='upper center',
                          )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
    else:
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.yaxis.grid(color="gray", alpha=0.5)
        ax.plot(x, y, 'o', label=label_orig, color=color, alpha=alpha)
        ax.plot(x, res.intercept + res.slope * x, 'r', label=label, color=color)
        xpoints = ypoints = plt.xlim()
        ax.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
        at = AnchoredText(f"R-squared: {res.rvalue ** 2:.2f}",
                          prop=dict(size=15), frameon=True,
                          loc='lower right',
                          )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
    plt.ylabel("matched wind speed m/s", fontsize=16)
    plt.xlabel("observed wind speed m/s", fontsize=16)
    plt.legend()
