from matplotlib import pyplot as plt
from readobs import load_observed_jul
from settings import START2, END2, STATION_NAMES_JUL

if __name__ == '__main__':
    # load: Grid, Catalogue, Observations
    observations = load_observed_jul(START2, END2)
    obs_wind = observations[:, 2, :]
    dir_wind = observations[:, 1, :]
    obs_u = observations[:, 3, :]
    obs_v = observations[:, 4, :]
    ncols = 2
    nrows = 7
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 15), sharex=True, sharey=True)

    k = 0
    for i in range(nrows):

        for j in range(ncols):
            try:
                axes[i, j].grid(True, linewidth=0.5, color="black", linestyle="-")
                axes[i, j].plot(observations[:, 1, k], label="wind observed", marker="", color="darkblue",
                                linestyle="--")
                axes[i, j].set_title(f"{STATION_NAMES_JUL[k]}")
                axes[i, j].legend(loc="upper right")

                axes[i, j].set_title(f"matched and observed wind for measuring station {STATION_NAMES_JUL[k]}",
                                     fontsize=24)
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
            plt.savefig(f"plots/observations.png", bbox_inches='tight')
    plt.figure()
    plt.show()
    plt.close()

    print("done.")
