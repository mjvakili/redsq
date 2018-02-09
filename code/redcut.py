import numpy as np
import cosmolopy
from scipy.interpolation import CubicSpline
import cosmolopy.distance as cd
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
    #for the sake of numerical stability we devide dvdz by 10000.0
    return dvdz

class cutter(object):


    def __init__(self, zmin, zmax, dz, Nf, cat):
 
        self.zmin = zmin #minimum redshift
        self.zmax = zmax #maximum redshift
        self.dz = dz     #width of redshift bins
	self.Nf = Nf 
        #initializing the nods where the maximum chisquared is defined
        self.nod = np.linspace(self.zmin, self.zmax,self.Nf)
        self.nod = .5 * (self.nod[:-1]+self.nod[1:]) 
        #initializing chi max values at the chimax nods
        self.chinod = 2.*np.ones_like((self.chinod))
        #data initialization
        
        self.opts, self.zs, self.chis, self.ls = \
        cat[:,0], cat[:,1], cat[:,2], cat[:,3]
       
        #only keeping the data for which optimization 
        #is successful
        self.zs = self.zs[self.opts==1]
        self.chis = self.chis[self.opts==1]
        self.ls = self.ls[self.opts==1]
        self.opts = self.opts[self.opts==1]

    def chiz(self, z):

        return CubicSpline(self.nod, self.chinod)(z)
     
    def reducer(self):

        mask = self.chis > self.chiz(self.zs)
        
        return cat[mask]

        

 if __name__ == '__main__':



   model = ezgal.model("/net/delft/data2/vakili/easy/ezgal_models/www.baryons.org/ezgal/models/bc03_burst_0.1_z_0.02_salp.model")
   zf = 3.0 #HARDCODED
   kcorr_sloan = model.get_kcorrects(zf=zf , zs = 0.2 , filters = "sloan_i")
   model.set_normalization("sloan_i" , 0.2 , 17.85  , vega=False, apparent=True)
   model.add_filter("/net/delft/data2/vakili/easy/i.dat" , "kids" , units = "nm")
   cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':1.0}
