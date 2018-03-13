import numpy as np
import pyfits as pf
import h5py
from astropy.coordinates import SkyCoord
from astropy import units as u


def match_2df_kids():

    '''
    function fo matching the sky 
    coordinate of 2dflens lrgs
    to their kids counterparts
    '''
    ### KiDS ###
    kids_cat = h5py.File("reduced_kids.h5" , 'r')
    kids_ra = kids_cat["RA"]
    kids_dec = kids_cat["DEC"]
    kids_id = kids_cat["ID"]
    
    ###2dfLenS LoZ & HiZ###

    df_cat_loz = np.loadtxt("data/data_loz_atlas_kidss_160105_ntar.dat")
    df_cat_hiz = np.loadtxt("data/data_hiz_atlas_kidss_160105_ntar.dat")
    

    dfn_cat_loz = np.loadtxt("data/data_loz_atlas_kidsn_160105_ntar.dat")
    dfn_cat_hiz = np.loadtxt("data/data_hiz_atlas_kidsn_160105_ntar.dat")

    
    loz_ra = df_cat_loz[:,0]
    hiz_ra = df_cat_hiz[:,0]
    loz_dec = df_cat_loz[:,1]
    hiz_dec = df_cat_hiz[:,1]
    loz_z = df_cat_loz[:,2]
    hiz_z = df_cat_hiz[:,2]
    
    nloz_ra = dfn_cat_loz[:,0]
    nhiz_ra = dfn_cat_hiz[:,0]
    nloz_dec = dfn_cat_loz[:,1]
    nhiz_dec = dfn_cat_hiz[:,1]
    nloz_z = dfn_cat_loz[:,2]
    nhiz_z = dfn_cat_hiz[:,2]

    df_z = np.hstack([loz_z, hiz_z, nloz_z, nhiz_z])
    df_ra = np.hstack([loz_ra, hiz_ra, nloz_ra, nhiz_ra])
    df_dec = np.hstack([loz_dec, hiz_dec, nloz_dec, nhiz_dec])

    c_kids = SkyCoord(ra=kids_ra*u.degree, dec=kids_dec*u.degree)   
    c_df = SkyCoord(ra=df_ra*u.degree, dec=df_dec*u.degree)   

    idx, d2d, d3d = c_df.match_to_catalog_sky(c_kids) 
    
    print "matching done"
    
    np.savetxt("2df_matched_kids.dat",np.vstack([idx, df_ra, df_dec, df_z]))

    return None

if __name__ == '__main__':

    match_2df_kids()
