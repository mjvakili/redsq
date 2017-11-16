import pyfits as pf
import numpy as np


def writer():

    a = pf.open("KiDS_DR3_GAMA-like_rAUTOless20.3_NOMASKING_forMV.fits")[1].data
    a = a[a['IMAFLAGS_ISO_G']&01010111==0]
    a = a[a['IMAFLAGS_ISO_U']&01010111==0]
    a = a[a['IMAFLAGS_ISO_R']&01010111==0]
    a = a[a['IMAFLAGS_ISO_I']&01010111==0]
    a = a[a['MAG_GAAP_g_CALIB']<25.6]
    a = a[a['MAG_GAAP_u_CALIB']<25.4]
    a = a[a['MAG_GAAP_i_CALIB']<24.5]
    a = a[a['MAG_GAAP_r_CALIB']<24.7]
    a = a[a['MAGERR_GAAP_G']>0]
    a = a[a['MAGERR_GAAP_U']>0]
    a = a[a['MAGERR_GAAP_R']>0]
    a = a[a['MAGERR_GAAP_I']>0]


    galid , ra , dec , zphot, zerr = np.arange(a.shape[0]) , a['RAJ2000'] , a['DECJ2000'] , a['zphot_ANNz2'] , 0.0203 * (1. +  a['zphot_ANNz2'])
    

    return galid , ra, dec, zphot, zerr



def finder():

    a = pf.open("KiDS_DR3_GAMA-like_rAUTOless20.3_NOMASKING_forMV.fits")[1].data
    a = a[a['IMAFLAGS_ISO_G']&01010111==0]
    a = a[a['IMAFLAGS_ISO_U']&01010111==0]
    a = a[a['IMAFLAGS_ISO_R']&01010111==0]
    a = a[a['IMAFLAGS_ISO_I']&01010111==0]
    a = a[a['MAG_GAAP_g_CALIB']<25.6]
    a = a[a['MAG_GAAP_u_CALIB']<25.4]
    a = a[a['MAG_GAAP_i_CALIB']<24.5]
    a = a[a['MAG_GAAP_r_CALIB']<24.7]
    a = a[a['MAGERR_GAAP_G']>0]
    a = a[a['MAGERR_GAAP_U']>0]
    a = a[a['MAGERR_GAAP_R']>0]
    a = a[a['MAGERR_GAAP_I']>0]

    ind = np.loadtxt("index.dat") - 1
    ind = ind.astype(int)
    print ind

    gr = a['MAG_GAAP_g_CALIB'][ind] - a['MAG_GAAP_r_CALIB'][ind]

    ra = a['RAJ2000'][ind] 
    dec = a['DECJ2000'][ind]
    z = a['zphot_ANNz2'][ind]

    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    plt.figure()                                                                            
    ax = plt.gca()
    ax.scatter(ra , dec , s = 1.0 , c = z)
    for i, txt in enumerate(z):
        ax.annotate(txt, (ra[i],dec[i]))

    plt.savefig("/home/vakili/public_html/files/example_group.png")
    #plt.figure(figsize = (5,5))                                                                            
    #ax = plt.gca()
    #ax.scatter(ra , dec , s = 1.0 , color = z)
    #plt.colorbar()
    #plt.savefig("/home/vakili/public_html/files/example_color.png")
    return None 


if __name__ == '__main__':


   finder()
   """
   x,y,z,w,v = writer()

   arr = np.empty((x.shape[0] , 5)) #, 'i8 , f8 , f8 , f8 , f8')
   
   arr[:,0] = x
   arr[:,1] = y
   arr[:,2] = z
   arr[:,3] = w
   arr[:,4] = v


   print arr
   #arr = np.vstack([x,y,z,w,v]).T

   #arr = np.array(arr , dtype = 'i8 , f8 , f8 , f8 , f8')

   #print arr.dtype

   np.savetxt("test_phot.txt" , arr)# , fmt = '%i, %f , %f , %f , %f')

   """
