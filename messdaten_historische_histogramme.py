import pandas as pd
from matplotlib import pyplot as plt

plt.style.use("bmh")

if __name__ == '__main__':
    df_wind = pd.read_csv("historischeDaten/BerlinerStraße/LUBW/berlinerstraße1320.csv",
                          usecols=["Windgeschwindigkeit (m/s)"])
    df_wind_dir = pd.read_csv("historischeDaten/BerlinerStraße/LUBW/berlinerstraße1320.csv",
                              usecols=["Windrichtung (°)"])
    df_wieb = pd.read_csv("historischeDaten/Wieblingen_West_1989-96.csv", usecols=["WiRi Grad", "WiGe m/s"],
                          skiprows=[i for i in range(1, 3045)])
    df_wieb_speed = pd.read_csv("historischeDaten/Wieblingen_West_1989-96.csv", usecols=["WiGe m/s"],
                                skiprows=[i for i in range(1,
                                                           3045)])
    indexNames = df_wieb_speed[(df_wieb_speed['WiGe m/s'] >= 14)].index
    df_wieb_speed.drop(indexNames, inplace=True)

    df_wieb_dir = pd.read_csv("historischeDaten/Wieblingen_West_1989-96.csv", usecols=["WiRi Grad"],
                              skiprows=[i for i in range(1,
                                                         3045)])
    df_eich_speed = pd.read_csv("historischeDaten/DreiEichen_1994-96.csv", usecols=[
        "WiGe m/s"])
    df_eich_dir = pd.read_csv("historischeDaten/DreiEichen_1994-96.csv", usecols=[
        "WiRi Grad"])
    indexNames = df_eich_speed[(df_eich_speed['WiGe m/s'] >= 14)].index
    df_eich_speed.drop(indexNames, inplace=True)

    df_geo_speed = pd.read_csv("historischeDaten/GeographischesInstitut_1989-95.csv", usecols=[
        "WiGe m/s"])
    indexNames = df_geo_speed[(df_geo_speed['WiGe m/s'] >= 14)].index
    df_geo_speed.drop(indexNames, inplace=True)

    df_geo_dir = pd.read_csv("historischeDaten/GeographischesInstitut_1989-95.csv", usecols=[
        "WiRi Grad"])
    df_koe_speed = pd.read_csv("historischeDaten/Koenigsstuhl/Koenigsstuhl_1989-93.csv", usecols=[
        "WiGe m/s"])
    indexNames = df_koe_speed[(df_koe_speed['WiGe m/s'] >= 14)].index
    df_koe_speed.drop(indexNames, inplace=True)

    df_koe_dir = pd.read_csv("historischeDaten/Koenigsstuhl/Koenigsstuhl_1989-93.csv", usecols=[
        "WiRi Grad"])

    vber = df_wind.to_numpy()
    vwieb = df_wieb_speed.to_numpy()
    veich = df_eich_speed.to_numpy()
    vgeo = df_geo_speed.to_numpy()
    vkoep = df_koe_speed.to_numpy()

    kwargs = dict(histtype="stepfilled", alpha=0.3, density=True, bins=50)

    plt.hist(vber, **kwargs, label="Berliner Straße")
    plt.hist(vwieb, **kwargs, label="Wieblingen")
    plt.hist(veich, **kwargs, label="Drei Eichen")
    plt.hist(vgeo, **kwargs, label="Geographisches Institut")
    plt.hist(vkoep, **kwargs, label="Köpfel")
    plt.legend()
    plt.title("frequency of wind speeds [m/s]")
    plt.xlabel("wind speed [m/s]")
    plt.ylabel("frequency")
    plt.savefig("plots/windspeed_historical.png", bbox_inches='tight')
    plt.show()
    plt.close()
    print("done.")
