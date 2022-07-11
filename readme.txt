pre-process and functions:
settings.py: Settings for matching, GRAL or GRAMM, which catalog, stations used, smoothing, selection of metric, period of time
plotting.py: Plot functions for time series matched and observerd wind fields
gral_cat_with_sb.py: adds to GRAL catalog SB GRAMM wind fields
math_functions.py: used math functions
match-obs-functions: Functions for matching
loadGRALwindfield.py: loads from .gff files wind fields at stations
readcat.py: reads GRAMM catalog
readobs.py: read observed data of all stations
readobsdigi.py: observed data of the stations of the digital agency
readobsiup.py: observed data of the IUP station
readobslubw.py: observed data of the LUBW station
readobsph: observed data of the PH stations

match-to-observation:
match-to-observation.py: performs matching, generates time series plot, calculation of RMSE, MB


data (on request):
historischeDaten: used historical data of Berlinerstraße, Königstuhl, Drei Eichen, Geo Institut, Wieblingen West
messdaten/utc: Measurement data used, created by: readobslubw.py, readobsnils.py, readobsdigi.py, readobsiup.py


plots and evaluation:

anisotrop_metric.py: Evaluation of the anisotropic metric for different lambda
anisotropicvis.py: Plot of the anisotropic metric for different lambda with synthetic data.
artificial_windrose.py: Wind roses of the catalog at all stations
catalog_sit_coice.py: Comparison of two catalogs, RMSE, MB, lin regression plot
correlations.py: Consideration of various correlations
flowfield_gral.py: Plot GRAL flowfield for specific situations
flowfield_gramm.py: Plot GRAMM flowfield for specific situations
groupings.py: simulated wind fields grouped by wind direction
logarithmic.py: Observation of the wind at different heights for selected stations
matching_one_stat.py: Match-to-observation performed at one station for all stations
messdaten_histogramme.py: Plot histograms for used measurement data
messdaten_hisotrische_histogramme.py: Plot histogram for historical data
metric_test.py: Compare different metrics with each other
observations.py: Measured wind speed of the individual stations
permutations.py: For number n of stations, all possible combinations of 14 stations
read_meteopgt.py: Evaluation of the weather situations used
rmse_catalog.py: Calculation and plotting of the RMSE vs. number of catalog entries
rmse_mean_bias.py: rmse vs mb calculation and plots, lin regression of stations
rmse_stations.py: Calculation and plotting of the mean RMSE vs. number of stations
rmse_stations_best.py: Calculation of min, max, mean RMSEs vs. number of stations
dailycycle.py: Daily cycle Observed wind speed, simulated wind speed for all stations
windrosen.py: Wind roses for observation, matching(GRAL, GRAMM) for all stations

.pkl files stored: https://heibox.uni-heidelberg.de/d/843f046b85e646b1a1c7/
	winds_cat1008_final.pkl: GRAL catalog for 1008 entries (entire domain)
	winds_cat1476_final.pkl: GRAL catalog for 1476 entries (entire domain)
	gff1008_final.pkl: GRAL catalog for 1008 entries (only all stations)
	gff1476_final.pkl: GRAL catalog for 1476 entries (only all stations)
you can download the files: wget -c https://heibox.uni-heidelberg.de/d/843f046b85e646b1a1c7/



environmental.yml: built environment with "conda env create -f environmental.yml"
