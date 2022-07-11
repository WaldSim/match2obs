LOAD_CATALOGUE = True
USE_WEIGHTING = False
USE_SMOOTHING = False
N_CATALOGUE_USED = 1476
#N_CATALOGUE_USED = 1008
METRIK = "l2vobsminmean"
GRAL = True
START2= "2021-07-22 00:00:00"
END2 = "2021-08-21 23:00:00"
#USED_STATIONS= [0,1, 2,3, 4,5,6, 7,8, 9, 10, 11, 12,13]
USED_STATIONS= [0,1, 2,3, 4,5,6,7, 8,9, 10, 11, 12, 13]
STATION_NAMES_JUL = {0: "GHW", 1: "HP", 2: 'KOS', 3: 'STW', 4: 'PT', 5: 'SB',
                     6: 'STB', 7: 'THB', 8: 'WWR', 9: 'KOE',10: 'CZE', 11:'GAB', 12:'LUBW', 13:'IUP'}
STATION_COLORS = {0: "maroon", 1: "darkorange", 2: 'gold', 3: 'darkolivegreen', 4: 'turquoise', 5: 'slategrey',
                  6: 'darkcyan', 7: 'magenta',
                  8: "blue", 9: "black", 10: "grey", 11: "yellow", 12: "green", 13: "pink"}
# Koordinaten, Messhöhe (für GRAL), Messhöhe + Gebäude (für GRAMM), Höhe über Grund + Gebäude
STATIONDATA_JUL = {"GHW": ["GHW1", "GHW2", 3472888.990, 5475770.394, 10, 10, 107],
               "HP": ["HP1", "HP2", 3477413.573, 5470824.362, 8,8, 113],
               "KOS" : ["KOS1", "KOS2",3480342.295, 5474125.002, 4,4, 561],
                "STW": ["STW1", "STW2", 3480078.171, 5473526.270, 10, 22, 569],
               "PT" : ["PT1", "PT2", 3482028.905,5479563.560, 4, 4, 351],
               "SB" : ["SB1","SB2", 3483945.193, 5474103.973, 4, 4, 119],
             "STB": ["STB1", "STB2", 3477329.729, 5474470.342, 10, 20,123],
               "THB": ["THB1","THB2", 3477783.586, 5475052.152, 16, 16 ,105],
                "WWR": ["WWR1", "WWR2", 3472735.935, 5476113.269, 10, 10, 107],
               "KOE": ["KOE1", "KOE2", 3481930.604, 5476005.216, 6 , 6, 223],
                 "CZE": ["CZE1", "CZE2", 3476559.336, 5473792.047, 10,28 , 131],
               "GAB": ["GAB1", "GAB2", 3481423.178, 5469356.975, 10, 10,321],
                   "LUBW": ["LUBW1", "LUBW2", 3476616.415, 5475898.736, 10, 10, 111],
                   "IUP":["IUP1","IUP2",3476454.968, 5475644.404, 6.3, 36.6, 143],
                   }
STATIONS_INDEX = {"GHW": 0 , "HP":1, 'KOS':2,'STW':3,'PT':4,
                    'SB':5,
                     'STB':6, 'THB':7,'WWR':8,  'KOE':9,'CZE':10,'GAB':11,'LUBW':12, 'IUP':13}



STATION_HEIGHTS = [117, 120, 565, 581, 353, 119, 134, 143, 116, 230, 150, 331, 122, 142]
