import pickle
import numpy as np
from einops import rearrange
from matchobs_functions import  cat_jul_1008, cat_jul_1476
from settings import LOAD_CATALOGUE
def matchgralorgramm_jul_1008(gral, load_cat):
    if gral:
        pkl_file = open('gff1008final.pkl', 'rb')
        catalogue = pickle.load(pkl_file)
        pkl_file.close()
    else:
        catalogue = cat_jul_1008(load_cat)
    return catalogue
def matchgralorgramm_jul_1476(gral, load_cat):
    if gral:
        pkl_file = open('gff1476final.pkl', 'rb')
        catalogue = pickle.load(pkl_file)
        pkl_file.close()
    else:
        catalogue = cat_jul_1476(load_cat)
    return catalogue
if __name__ == '__main__':
    GRAL = True
    catalogue_gral = matchgralorgramm_jul_1476(GRAL, LOAD_CATALOGUE)
    z_layer = [11,  13, 235, 242, 130,  19,  13,  11,  67,  23, 118,  13,  27] # neu gewählte zlayer 17.02.22, "winds_cat1476_final.pkl"
    cat_gral = []
    for i in range(0, len(z_layer)):
        cat = catalogue_gral[:, i, z_layer[i], :]
        cat_gral.append(cat)
    cat_gral = np.array(cat_gral)
    cat_gral = rearrange(cat_gral, "s c d   -> d s c")
    GRAL = False
    catalogue = matchgralorgramm_jul_1476(GRAL, LOAD_CATALOGUE)
    # SB is 5th Station
    cat_gramm = catalogue[:, 5, :]
    cat_gramm = cat_gramm[:, None, :]
    cat_gral_1 = cat_gral[:, 0:5, :]
    cat_gral_2 = cat_gral[:, 5::, :]
    new_catalogue = np.concatenate((cat_gral_1, cat_gramm, cat_gral_2), axis=1)
    output = open('winds_cat_1476_with_sb_test.pkl', 'wb') # name new catalogc
    pickle.dump(new_catalogue, output)
    output.close()
    new_catalogue = np.asarray(new_catalogue)
    # nans out
    invalids = np.argwhere(np.isnan(new_catalogue[0, 0, :]))
    invalids = np.concatenate(invalids).ravel().tolist()
    np.save("invalids1476_test.npy", invalids)
    new_indices = []
    for element in range(1476):
        if element not in invalids:
            new_indices.append(element)
    cat_final  =new_catalogue[:, :, new_indices]
    print("done.")
