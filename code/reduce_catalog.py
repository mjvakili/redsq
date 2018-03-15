import pyfits as pf
import h5py
import numpy as np



def reduce_catalog():

    data = pf.open("data/KiDS_DR3_merged_FULL_wihANNz2_forJonas.fits")[1].data
    data = data[(data['IMAFLAGS_ISO_U']&01010111)== 0]
    data = data[(data['IMAFLAGS_ISO_G']&01010111)== 0]
    data = data[(data['IMAFLAGS_ISO_R']&01010111)== 0]
    data = data[(data['IMAFLAGS_ISO_I']&01010111)== 0]
    data = data[data['MAGERR_GAAP_U'] >0]
    data = data[data['MAGERR_GAAP_G'] >0]
    data = data[data['MAGERR_GAAP_R'] >0]
    data = data[data['MAGERR_GAAP_I'] >0]
    #data = data[data['CLASS_STAR']<0.8]
    #data = data[data['SG2DPHOT']==0]
    CLASS_STAR = data['CLASS_STAR']
    SG2PHOT = data['SG2DPHOT']
    print "passed star-galaxy separation!"
    ug = data['MAG_GAAP_u_CALIB'] - data['MAG_GAAP_g_CALIB']
    gr = data['MAG_GAAP_g_CALIB'] - data['MAG_GAAP_r_CALIB']
    ri = data['MAG_GAAP_r_CALIB'] - data['MAG_GAAP_i_CALIB']

    colors = np.vstack([ug,gr,ri]).T
    mi = data['MAG_GAAP_i_CALIB']

    color_errs = np.zeros((colors.shape[0], colors.shape[1], colors.shape[1]))
    color_errs[:,0,0] = data['MAGERR_GAAP_U']**2 + data['MAGERR_GAAP_G']**2
    color_errs[:,1,1] = data['MAGERR_GAAP_G']**2 + data['MAGERR_GAAP_R']**2
    color_errs[:,2,2] = data['MAGERR_GAAP_R']**2 + data['MAGERR_GAAP_I']**2
    color_errs[:,0,1] = -1. * data['MAGERR_GAAP_G']**2
    color_errs[:,1,0] = -1. * data['MAGERR_GAAP_G']**2
    color_errs[:,1,2] = -1. * data['MAGERR_GAAP_R']**2
    color_errs[:,2,1] = -1. * data['MAGERR_GAAP_R']**2

    RA, DEC = data['RAJ2000'], data['DECJ2000']
    ID = data['ID']
    redshift = data['zphot_ANNz2']

    result_file = h5py.File("reduced_kids.h5" , 'w')
    Ngals = mi.shape[0]
    result_file.create_dataset("ID" , (Ngals,  ) , data = ID, dtype = 'S25')
    result_file.create_dataset("RA" , (Ngals,  ) , data = RA)
    result_file.create_dataset("DEC", (Ngals,  ) , data = DEC)
    result_file.create_dataset("mi", (Ngals, ) , data = mi)
    result_file.create_dataset("redshift", (Ngals, ) , data = redshift)
    result_file.create_dataset("colors", (Ngals, 3) , data = colors)
    result_file.create_dataset("color_errs", (Ngals, 3, 3) , data = color_errs)
    result_file.create_dataset("CLASS_STAR", (Ngals, ) , data = CLASS_STAR)
    result_file.create_dataset("SG2PHOT", (Ngals, ) , data = SG2PHOT)
   
    result_file.close()

    return None

def sdss_reduction():
    
    gama = pf.open("data/KiDSxGAMAequ+G23.fits")[1].data
    sdss = pf.open("data/KiDS.DR3.x.SDSS.DR13.galaxies.fits")[1].data
    mask = np.where(np.in1d(sdss['ID'] , gama['ID'])==False)
    sdss = sdss[mask]
    data = sdss
    data = data[data['IMAFLAGS_ISO_U']&01010111==0]
    data = data[data['IMAFLAGS_ISO_G']&01010111==0]
    data = data[data['IMAFLAGS_ISO_R']&01010111==0]
    data = data[data['IMAFLAGS_ISO_I']&01010111==0]
    data = data[data['MAGERR_GAAP_U']>0]
    data = data[data['MAGERR_GAAP_G']>0]
    data = data[data['MAGERR_GAAP_R']>0]
    data = data[data['MAGERR_GAAP_I']>0]
    u = data['MAG_GAAP_U'] + data['ZPT_OFFSET_U'] - data['EXT_SFD_U']
    g = data['MAG_GAAP_G'] + data['ZPT_OFFSET_G'] - data['EXT_SFD_G']
    r = data['MAG_GAAP_R'] + data['ZPT_OFFSET_R'] - data['EXT_SFD_R']
    i = data['MAG_GAAP_I'] + data['ZPT_OFFSET_I'] - data['EXT_SFD_I']
    mask = (u<25.4)&(g<25.6)&(r<24.7)&(i<24.5)
    data = data[mask]
    u = data['MAG_GAAP_U'] + data['ZPT_OFFSET_U'] - data['EXT_SFD_U']
    g = data['MAG_GAAP_G'] + data['ZPT_OFFSET_G'] - data['EXT_SFD_G']
    r = data['MAG_GAAP_R'] + data['ZPT_OFFSET_R'] - data['EXT_SFD_R']
    i = data['MAG_GAAP_I'] + data['ZPT_OFFSET_I'] - data['EXT_SFD_I']
    uerr = data['MAGERR_GAAP_U']
    gerr = data['MAGERR_GAAP_G']
    rerr = data['MAGERR_GAAP_R']
    ierr = data['MAGERR_GAAP_I']
    Z = data['Z']
    RA = data['RA']
    DEC = data['DEC']
    u_g = data['COLOR_GAAPHOM_U_G']
    g_r = data['COLOR_GAAPHOM_G_R']
    r_i = data['COLOR_GAAPHOM_R_I']
    col = np.vstack([u,g,r,i,uerr,gerr,rerr,ierr,u_g,g_r,r_i,Z,RA,DEC])

    return col

def gama_reduction():

    gama = pf.open("data/KiDSxGAMAequ+G23.fits")[1].data
    data = gama
        
    data = data[data['IMAFLAGS_ISO_U']&01010111==0]
    data = data[data['IMAFLAGS_ISO_G']&01010111==0]
    data = data[data['IMAFLAGS_ISO_R']&01010111==0]
    data = data[data['IMAFLAGS_ISO_I']&01010111==0]			    
    data = data[data['MAGERR_GAAP_U']>0]
    data = data[data['MAGERR_GAAP_G']>0]
    data = data[data['MAGERR_GAAP_R']>0]
    data = data[data['MAGERR_GAAP_I']>0]
    data = data[data['MAG_GAAP_u_CALIB']<25.4]
    data = data[data['MAG_GAAP_g_CALIB']<25.6]
    data = data[data['MAG_GAAP_r_CALIB']<24.7]
    data = data[data['MAG_GAAP_i_CALIB']<24.5]
    u = data['MAG_GAAP_u_CALIB']
    g = data['MAG_GAAP_g_CALIB']									        
    r = data['MAG_GAAP_r_CALIB']										    
    i = data['MAG_GAAP_i_CALIB']
    uerr = data['MAGERR_GAAP_U']
    gerr = data['MAGERR_GAAP_G']
    rerr = data['MAGERR_GAAP_R']
    ierr = data['MAGERR_GAAP_I']
    Z = data['Z']
    RA = data['RA']
    DEC = data['DEC']
    u_g = data['COLOR_GAAPHOM_U_G']
    g_r = data['COLOR_GAAPHOM_G_R']
    r_i = data['COLOR_GAAPHOM_R_I']
    col = np.vstack([u,g,r,i,uerr,gerr,rerr,ierr,u_g,g_r,r_i,Z,RA,DEC])													    
    return col 

def cosmos_reduction():
    
    cosmos = pf.open("data/KiDS.x.zCOSMOS.fits")[1].data
    data = cosmos

    data = data[data['NIMAFLAGS_ISO_THELI']&01010111==0]

    data = data[data['MAGERR_GAAP_U']>0]
    data = data[data['MAGERR_GAAP_G']>0]
    data = data[data['MAGERR_GAAP_R']>0]
    data = data[data['MAGERR_GAAP_I']>0]
    
    data = data[data['MAG_GAAP_u_CALIB']<25.4]
    data = data[data['MAG_GAAP_g_CALIB']<25.6]
    data = data[data['MAG_GAAP_r_CALIB']<24.7]
    data = data[data['MAG_GAAP_i_CALIB']<24.5]
    
    u = data['MAG_GAAP_u_CALIB']
    g = data['MAG_GAAP_g_CALIB']
    r = data['MAG_GAAP_r_CALIB']
    i = data['MAG_GAAP_i_CALIB']
    
    uerr = data['MAGERR_GAAP_U']
    gerr = data['MAGERR_GAAP_G']
    rerr = data['MAGERR_GAAP_R']
    ierr = data['MAGERR_GAAP_I']

    Z = data['Zspec']
    RA = data['RA_THELI']
    DEC = data['DEC_THELI']

    u_g = data['MAG_GAAP_u_CALIB'] - data['MAG_GAAP_g_CALIB']
    g_r = data['MAG_GAAP_g_CALIB'] - data['MAG_GAAP_r_CALIB']
    r_i = data['MAG_GAAP_r_CALIB'] - data['MAG_GAAP_i_CALIB']

    col = np.vstack([u,g,r,i,uerr,gerr,rerr,ierr,u_g,g_r,r_i,Z,RA,DEC])
    
    return col 

def deep_reduction():

    deep = pf.open("data/KiDS-like.x.DEEP2.DR4.fits")[1].data
    data = deep
    data = data[data['NIMAFLAGS_ISO_THELI']&01010111==0]
    data = data[data['MAGERR_GAAP_U']>0]
    data = data[data['MAGERR_GAAP_G']>0]
    data = data[data['MAGERR_GAAP_R']>0]
    data = data[data['MAGERR_GAAP_I']>0]
    
    data = data[data['MAG_GAAP_u_CALIB']<25.4]
    data = data[data['MAG_GAAP_g_CALIB']<25.6]
    data = data[data['MAG_GAAP_r_CALIB']<24.7]
    data = data[data['MAG_GAAP_i_CALIB']<24.5]
    
    u = data['MAG_GAAP_u_CALIB']
    g = data['MAG_GAAP_g_CALIB']
    r = data['MAG_GAAP_r_CALIB']
    i = data['MAG_GAAP_i_CALIB']
    
    uerr = data['MAGERR_GAAP_U']
    gerr = data['MAGERR_GAAP_G']
    rerr = data['MAGERR_GAAP_R']
    ierr = data['MAGERR_GAAP_I']

    Z = data['Zspec']
    RA = data['RA_THELI']
    DEC = data['DEC_THELI']

    u_g = data['MAG_GAAP_u_CALIB'] - data['MAG_GAAP_g_CALIB']
    g_r = data['MAG_GAAP_g_CALIB'] - data['MAG_GAAP_r_CALIB']
    r_i = data['MAG_GAAP_r_CALIB'] - data['MAG_GAAP_i_CALIB']

    col = np.vstack([u,g,r,i,uerr,gerr,rerr,ierr,u_g,g_r,r_i,Z,RA,DEC])
    
    return col


def reduce_spec_catalog():

    cat = np.hstack([gama_reduction() , sdss_reduction()]).T
        
    mi = cat[:,3]
    ug = cat[:,8]
    gr = cat[:,9]
    ri = cat[:,10]
    uerr , gerr , rerr , ierr = cat[:,4], cat[:,5], cat[:,6], cat[:,7]
    
    colors = np.vstack([ug,gr,ri]).T
    color_errs = np.zeros((colors.shape[0], colors.shape[1], colors.shape[1]))
    color_errs[:,0,0] = uerr**2. + gerr**2.
    color_errs[:,1,1] = gerr**2. + rerr**2.
    color_errs[:,2,2] = rerr**2. + ierr**2.
    color_errs[:,0,1] = -1. * gerr**2
    color_errs[:,1,0] = -1. * gerr**2
    color_errs[:,1,2] = -1. * rerr**2
    color_errs[:,2,1] = -1. * rerr**2

    RA, DEC = cat[:,12], cat[:,13]
    redshift = cat[:,11]

    result_file = h5py.File("reduced_speckids.h5" , 'w')
    Ngals = mi.shape[0]
    #result_file.create_dataset("ID" , (Ngals,  ) , data = ID, dtype = 'S25')
    result_file.create_dataset("RA" , (Ngals,  ) , data = RA)
    result_file.create_dataset("DEC", (Ngals,  ) , data = DEC)
    result_file.create_dataset("mi", (Ngals, ) , data = mi)
    result_file.create_dataset("redshift", (Ngals, ) , data = redshift)
    result_file.create_dataset("colors", (Ngals, 3) , data = colors)
    result_file.create_dataset("color_errs", (Ngals, 3, 3) , data = color_errs)
    result_file.close()

    return None

def reduce_specall_catalog():

    cat = np.hstack([gama_reduction() , sdss_reduction() , cosmos_reduction(), deep_reduction()]).T
        
    mi = cat[:,3]
    ug = cat[:,8]
    gr = cat[:,9]
    ri = cat[:,10]
    uerr , gerr , rerr , ierr = cat[:,4], cat[:,5], cat[:,6], cat[:,7]
    
    colors = np.vstack([ug,gr,ri]).T
    color_errs = np.zeros((colors.shape[0], colors.shape[1], colors.shape[1]))
    color_errs[:,0,0] = uerr**2. + gerr**2.
    color_errs[:,1,1] = gerr**2. + rerr**2.
    color_errs[:,2,2] = rerr**2. + ierr**2.
    color_errs[:,0,1] = -1. * gerr**2
    color_errs[:,1,0] = -1. * gerr**2
    color_errs[:,1,2] = -1. * rerr**2
    color_errs[:,2,1] = -1. * rerr**2

    RA, DEC = cat[:,12], cat[:,13]
    redshift = cat[:,11]

    result_file = h5py.File("reduced_speckids.h5" , 'w')
    Ngals = mi.shape[0]
    #result_file.create_dataset("ID" , (Ngals,  ) , data = ID, dtype = 'S25')
    result_file.create_dataset("RA" , (Ngals,  ) , data = RA)
    result_file.create_dataset("DEC", (Ngals,  ) , data = DEC)
    result_file.create_dataset("mi", (Ngals, ) , data = mi)
    result_file.create_dataset("redshift", (Ngals, ) , data = redshift)
    result_file.create_dataset("colors", (Ngals, 3) , data = colors)
    result_file.create_dataset("color_errs", (Ngals, 3, 3) , data = color_errs)
    result_file.close()

    return None

if __name__ == '__main__':


   #reduce_catalog()
   reduce_spec_catalog()
   reduce_specall_catalog()
