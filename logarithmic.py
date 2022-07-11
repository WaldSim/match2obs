import pickle
import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt

if __name__ == '__main__':
    pkl_file = open('gff1476final.pkl', 'rb')
    # pkl_file = open('gff1008final.pkl', 'rb')
    winds = pickle.load(pkl_file)
    ###z_layer = [8, 13, 234, 129, 12, 67, 117, 14, 233, 11, 24, 13, 13]
    sit = 1006 # Selects weather situation to be considered
    STATION = ["GHW", "HP", "KOS", "STW", "PT", "SB", "STB",
               "THB", "WWR", "KOE", "CZE", "GAB", "LUBW", "IUP"]

    # layer= [5, 8,228, 234, 135,  13, 6,5,61,13,113,13,12] # ground layer stationen
    layer = [6, 9, 233, 237, 128, 14, 5, 6, 64, 18, 113, 8, 24]
    winds_u = winds[sit, :, 0]
    winds_v = winds[sit, :, 1]
    winds_u_lubw = winds[sit, 12, layer[12]:, 0]
    winds_v_lubw = winds[sit, 12, layer[12]:, 1]
    winds_hor = np.sqrt(winds[sit, :, :, 0] ** 2 + winds[sit, :, :, 1] ** 2)
    winds_hor_2 = rearrange(winds_hor, "d s  -> s d ")
    winds_hor_trans = winds_hor.T

    plt.plot(winds_hor[0, layer[0]:], label=f"wind speed GHW")
    plt.title("")
    plt.xlabel("horizonal layer")
    plt.ylabel("wind speed m/s")
    plt.legend()
    plt.title("wind profile")
    plt.savefig("plots/winds_z_layer_GRAL_GHW_sit160.png", bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(winds_hor[12, layer[12]:], label=f"wind speed IUP")
    plt.xlabel("horizonal layer")
    plt.ylabel("wind speed m/s")
    plt.legend()
    plt.title("wind profile")
    plt.xlim(0, 40)
    plt.savefig("plots/winds_z_layer_GRAL_IUP_sit160.png", bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(winds_hor[11, layer[11]:], label=f"wind speed LUBW")
    plt.xlabel("horizonal layer")
    plt.ylabel("wind speed m/s")
    plt.legend()
    plt.title("wind profile")
    plt.savefig("plots/winds_z_layer_GRAL_LUBW_sit160.png", bbox_inches='tight')
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    x = np.arange(0, 20)
    for i in range(0, 12):
        plt.plot(x, (winds_hor[i, layer[i]:(20 + layer[i])]), label=f"wind speed {STATION[i]}", linestyle="-")
    plt.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
    plt.tight_layout()
    plt.title("")
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    plt.xlabel("height")
    plt.ylabel("wind speed [m/s]")
    plt.savefig("plots/windspeed_height.png")
    plt.show()


    print("done.")
