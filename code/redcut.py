import numpy as np
import cosmolopy
from scipy.interpolation import CubicSpline
import cosmolopy.distance as cd
import h5py
import scipy.optimize as op
import h5py

def dvdz(z):
    '''
    dv/dz to impose uniformity
    in redshift
    '''
    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72}
    d_a = cd.angular_diameter_distance(z, **cosmo) #angular diameter distance
    h = cd.e_z(z, **cosmo) #hubble parameter
    dvdz = (1+z)**2. * d_a **2  * h **-1. #dv/dz
    return dvdz

class cutter(object):


    def __init__(self, zmin, zmax, dz, Nf, cat,
                 chimax_glob, chimax_lbnd, 
		 chimax_ubnd, niter, nbins):
 
        self.zmin = zmin #minimum redshift
        self.zmax = zmax #maximum redshift
        self.dz = dz     #width of redshift bins
	self.Nf = Nf 
	self.chimax_glob = chimax_glob
	self.chimax_lbnd = chimax_lbnd
	self.chimax_ubnd = chimax_ubnd
        self.niter = niter
	self.nbins = nbins
        #initializing the nods where the maximum chisquared is defined
        self.nod = np.linspace(self.zmin, self.zmax,self.Nf)
        self.nod = .5 * (self.nod[:-1]+self.nod[1:]) 
        #initializing chi max values at the chimax nods
        self.chinod = 2.*np.ones_like((self.chinod))
        #data initialization
        self.cat = cat
        #only keeping the data for which optimization 
        #is successful
        #only keeping the candidates for which chi is less than 
        #the global maximum chi squared
        self.cat = self.cat[(self.cat[:,0]==1)&(self.cat[:,1]>self.zmin)& \
                            (self.cat[:,1]<self.zmax)&(self.cat[:,2]<self.chimax_global)]
        #initializing nods and maximum chis at nods			    
        self.nod = np.linspace(elf.zmin, self.zmax, self.Nf)
        self.chinods_init = 1.5 + np.zeros(self.Nf)
        self.chinods_init[::2] -= 0.5
        self.chinods = self.chinods_init.copy()

	return None

    def lnlike_chi(self, theta):

        chinods , norm = np.exp(theta[:-1]), np.exp(theta[-1])
	chi_maxs = CubicSpline(self.nods, chinods)(self.cat[:,1])
	cat_red = self.cat[self.cat[:,2] < chi_maxs]
	hist , edges = np.histogram(cat_red[:,1], bins = self.nbins)
        bins = .5*(edges[1:]+edges[:-1])
        dbin = edges[1] - edges[0]
        dvdbin = dvdz(bins)
        dvbin = dvdbin * dbin
        chisq = np.sum((hist - norm*dvbin)**2./(hist+ norm*dvbin)) 
        print chisq
        
	return chisq
    
    def lnlike_z(self):

        return None 

    def calibrator(self):


        return None

 if __name__ == '__main__':



   model = ezgal.model("/net/delft/data2/vakili/easy/ezgal_models/www.baryons.org/ezgal/models/bc03_burst_0.1_z_0.02_salp.model")
   zf = 3.0 #HARDCODED
   kcorr_sloan = model.get_kcorrects(zf=zf , zs = 0.2 , filters = "sloan_i")
   model.set_normalization("sloan_i" , 0.2 , 17.85  , vega=False, apparent=True)
   model.add_filter("/net/delft/data2/vakili/easy/i.dat" , "kids" , units = "nm")
   cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':1.0}
