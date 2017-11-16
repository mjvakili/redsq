import pyfits as pf
import matplotlib.pyplot as plt
import multiprocessing
import emcee
import numpy as np
plt.switch_backend("Agg")
import pandas as pd
import seaborn as sns 
import itertools
sns.set_style("white")
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
sns.set_palette(sns.color_palette(["#9b59b6", "#95a5a6", 
                                   "#e74c3c", "#3498db", 
				   "#34495e", "#2ecc71"]))
from astroML.density_estimation import XDGMM
from astroML.density_estimation import XDGMM
from matplotlib.patches import Ellipse
from astroML.plotting.tools import draw_ellipse
import multiprocessing
from multiprocessing import Pool

def filter(style):
    '''
    There are two options for filenames: KiDSxGAMA matched catalog
                                         KiDS GAMA-like sample
    filename = "KiDSxGAMAequ+G23.fits
    filename = "KiDS_DR3_GAMA-like_rAUTOless20.3_NOMASKING_forMV.fits"
    '''
    if style == 'GAMA-MATCHED': filename = "KiDSxGAMAequ+G23.fits"
    elif style == 'GAMA-LIKE' : filename = "KiDS_DR3_GAMA-like_rAUTOless20.3_NOMASKING_forMV.fits"

    a = pf.open(filename)[1].data
    
    a = a[a['IMAFLAGS_ISO_U']&01010111==0]
    a = a[a['IMAFLAGS_ISO_G']&01010111==0]
    a = a[a['IMAFLAGS_ISO_R']&01010111==0]
    a = a[a['IMAFLAGS_ISO_I']&01010111==0]

    a = a[a['MAGERR_GAAP_U']>0]
    a = a[a['MAGERR_GAAP_G']>0]
    a = a[a['MAGERR_GAAP_R']>0]
    a = a[a['MAGERR_GAAP_I']>0]

    a = a[a['MAG_GAAP_u_CALIB']<25.4]
    a = a[a['MAG_GAAP_g_CALIB']<25.6]
    a = a[a['MAG_GAAP_r_CALIB']<24.7]
    a = a[a['MAG_GAAP_i_CALIB']<24.5]

    return a

def extractor(zmin , zmax):
    '''
    test returns x = mi 
                 y = g-r
              yerr = (g-r)-err

    update :  1) make this return u-g , g-r , r-i
              1) this will require covariance between color components.
              2) return xerr = mi_err. Is it even worth it to include x errors?
    '''

    gals = pf.open("groups/G3CGalv07.fits")[1].data   #galaxy group catalog
    gals = gals[gals['GroupID']!=0]              #keeping galaxies that are in groups  
    match = filter('GAMA-MATCHED') 
    
    mask = np.in1d(match['CATAID'] , gals[gals['GroupID']!=0]['CATAID'])
    matched_gals = match[mask]

    ug = matched_gals['COLOR_GAAPHOM_U_G'] 
    gr = matched_gals['COLOR_GAAPHOM_G_R'] 
    ri = matched_gals['COLOR_GAAPHOM_R_I']
    z = matched_gals['Z']
    mi = matched_gals['MAG_GAAP_i_CALIB']
    colors = np.array([ug , gr , ri]).T
    

    u_err = matched_gals['MAGERR_GAAP_U']
    g_err = matched_gals['MAGERR_GAAP_G']
    i_err = matched_gals['MAGERR_GAAP_I']
    r_err = matched_gals['MAGERR_GAAP_R']
   
    Ngals = u_err.shape[0]
    c_err = np.zeros((Ngals , 3 , 3))
    c_err[:,0,0] = u_err**2. + g_err**2.
    c_err[:,1,1] = g_err**2. + r_err**2.
    c_err[:,2,2] = r_err**2. + i_err**2.
    c_err[:,0,1] = -1.*g_err**2.
    c_err[:,1,2] = -1.*r_err**2.
    c_err[:,1,0] = c_err[:,0,1]
    c_err[:,2,1] = c_err[:,1,2]

    redshift_mask = np.where((zmin<z)&(z<zmax))[0]
    colors = colors[redshift_mask , :]
    c_err = c_err[redshift_mask , : , :]
    
    return colors , c_err

def compute_XD_results(y , yerr, n_components=10, n_iter=500):
       
    clf = XDGMM(n_components, n_iter=n_iter)
    clf.fit(y , yerr)

    return clf

def covint(p):

   zmin , zmax = p
   X , Xerr = extractor(zmin , zmax) 	
   filename = "results3/gr_result_z_"+str(zmin)+"_"+str(zmax)+".txt"
   pmem = np.loadtxt(filename)[12:]

   y , yerr = X[pmem>0.8,:] , Xerr[pmem>0.8,:] 
   ncomp = 1
   clf = compute_XD_results(y, yerr, n_components = ncomp, n_iter=50)
   
   #print clf.mu , clf.V

   palette = itertools.cycle(sns.color_palette())
   fig, axs = plt.subplots(1, 2, figsize=(10.5, 5) , sharex = True, sharey = True)
   axs[0].scatter(X[:,0] , X[:,1] , s = 0.5 , c = next(palette), label = str(zmin)+"<$z$<"+str(zmax))
   axs[0].set_xlabel(r"$u-g$")
   axs[0].set_ylabel(r"$g-r$")
   axs[0].legend(loc = 'best', fontsize = 10)
   axs[1].scatter(X[pmem>0.8,0] , X[pmem>0.8,1] , s = 0.5 , c = next(palette) , label = r"$p_{\mathrm{red}}>0.8$")
   axs[1].set_xlabel(r"$u-g$")
   
   for i in range(ncomp):
       draw_ellipse(clf.mu[i][:2], clf.V[i][:2,:2], scales=[2], ax=axs[1],
                        ec='k', fc='blue', alpha=0.2 )
   
   plt.legend(loc = 'best' , fontsize = 10)
   fig.tight_layout()
   plt.savefig("/home/vakili/public_html/files/covariance/covariance_80_ug_gr_"+str(zmin)+"_"+str(zmax)+".png")

   palette = itertools.cycle(sns.color_palette())
   fig, axs = plt.subplots(1, 2, figsize=(10.5, 5), sharex = True, sharey = True)
   axs[0].scatter(X[:,1] , X[:,2] , s = 0.5 , c = next(palette), label = str(zmin)+"<$z$<"+str(zmax))
   axs[0].set_xlabel(r"$g-r$")
   axs[0].set_ylabel(r"$r-i$")
   axs[0].legend(loc = 'best', fontsize = 10)


   axs[1].scatter(X[pmem>0.8,1] , X[pmem>0.8,2] , s = 0.5 , c = next(palette), label = r"$p_{\mathrm{red}}>0.8$")
   axs[1].set_xlabel(r"$g-r$")
   for i in range(ncomp):
       draw_ellipse(clf.mu[i][1:], clf.V[i][1:,1:], scales=[2], ax=axs[1],
                        ec='k', fc='blue', alpha=0.2)
   plt.legend(loc = 'best' , fontsize = 10)
   fig.tight_layout()
   plt.savefig("/home/vakili/public_html/files/covariance/covariance_80_gr_ri_"+str(zmin)+"_"+str(zmax)+".png")
   
   
   palette = itertools.cycle(sns.color_palette())
   fig, axs = plt.subplots(1, 2, figsize=(10.5, 5), sharex = True, sharey = True)
   axs[0].scatter(X[:,0] , X[:,2] , s = 0.5 , c = next(palette), label = str(zmin)+"<$z$<"+str(zmax))
   axs[0].set_xlabel(r"$u-g$")
   axs[0].set_ylabel(r"$r-i$")
   axs[0].legend(loc = 'best', fontsize = 10)

   axs[1].scatter(X[pmem>0.8,0] , X[pmem>0.8,2] , s = 0.5 , c = next(palette), label = r"$p_{\mathrm{red}}>0.8$")
   axs[1].set_xlabel(r"$u-g$")
  
   for i in range(ncomp):
       draw_ellipse(np.delete(clf.mu[i],1,0) , np.delete(np.delete(clf.V[i],1,1),1,0) , scales=[2], ax=axs[1],
                        ec='k', fc='blue', alpha=0.2)
   plt.legend(loc = 'best' , fontsize = 10)
   fig.tight_layout()
   plt.savefig("/home/vakili/public_html/files/covariance/covariance_80_ug_ri_"+str(zmin)+"_"+str(zmax)+".png")

   return np.array(clf.V[0])   

if __name__ == '__main__':
   
   z_init = 0.2 
   Nthreads  = 10
   pool = Pool(Nthreads)
   mapfn = pool.map
   arglist = [None] * Nthreads
   
   for i in range(Nthreads):

       zmin = z_init + i * 0.02
       zmax = zmin + 0.02
       arglist[i] = (zmin, zmax)  
   
   result = list(mapfn(covint, [ars for ars in arglist]))
 
   for t in range(Nthreads):
       zmin = z_init + t * 0.02
       zmax = zmin + 0.02
       np.savetxt("results4/covariance_z_"+str(zmin)+"_"+str(zmax)+".txt" , np.array(result[t]))
   pool.close()
