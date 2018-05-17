'''
class for estimating
p(z|m,c) \propto p(c|m,z)p(m|z)p(z)
'''
import numpy as np
#import matplotlib.pyplot as plt
#plt.switch_backend("Agg")
import h5py
from scipy.interpolate import CubicSpline
import ezgal
import cosmolopy.distance as cd
#import util
#import kids_gama
import emcee
#from mixture_filtering import catalog_combinator
#import pyfits as pf

def lnprior(z, zmin , zmax):
    '''
    dv/dz to impose uniformity
    in redshift
    '''
    if z<0.01:
       return -np.inf
    elif z>2.0:
       return -np.inf
    else:   
       d_a = cd.angular_diameter_distance(z, **cosmo) #angular diameter distance
       h = cd.e_z(z, **cosmo) #hubble parameter
       dvdz = (1+z)**2. * d_a **2  * h **-1. #dv/dz
       #for the sake of numerical stability we devide dvdz by 10000.0
       return np.log(dvdz / 10000.0)

def redmapper_mstar(z):
    '''
    Rykoff+12 fitting formula for characteristic sloan iband mags 
    of redmapper galaxy cluster members at z<0.4.
    Used to normalize the Ezgal model
    '''
    return 22.44 + 3.36*np.log(z) + 0.273*np.log(z)**2 - 0.0618*np.log(z)**3 - 0.0227*np.log(z)**4

def kids_mstar(zs):
    '''
    returns mstar of redsequence galaxies 
    as observed by kids i band
    '''
    #kcorr = model.get_kcorrects(zf=3.0, zs=zs , filters = "kids") #WE SHALL KCORRECT THE MAGS IF NEEDED
    mags = model.get_apparent_mags(zf=zf , filters = "kids" , zs= zs)

    return mags

def schecter(m,z):
    '''
    magnitude distribution
    as a function of redshift
    '''
    mchar = kids_mstar(z)
    dm = m - mchar
    exparg = 10. ** (-0.4 * dm)

    return np.log((exparg ** 2.) * np.exp(-1.*exparg)) 

def luminosity(m,z):
    '''
    L/Lstar of redsequence galaxies
    '''
    mchar = kids_mstar(z)
    dm = m - mchar
    exparg = 10. ** (-0.4 * dm)

    return exparg

class estimate(object):

    def __init__(self, zmin, zmax, dz, Nb, Nm, Nf):
 
        self.zmin = zmin #minimum redshift
        self.zmax = zmax #maximum redshift
        self.dz = dz     #width of redshift bins
        self.Nb = Nb
	self.Nm = Nm
	self.Nf = Nf
        
        self.bnod = np.linspace(self.zmin,self.zmax,self.Nb) #spacing of .05
        self.mnod = np.linspace(self.zmin,self.zmax,self.Nm) #spacing of .1
        self.fnod = np.linspace(self.zmin,self.zmax,self.Nf) #spacing of .14
        self.xrefnod = np.linspace(self.zmin,self.zmax,20) #spacing of .05 
        self.bnod = .5*(self.bnod[1:]+self.bnod[:-1])
        self.fnod = .5*(self.fnod[1:]+self.fnod[:-1])
        self.mnod = .5*(self.mnod[1:]+self.mnod[:-1])

        self.theta = np.loadtxt("opt_theta_bfgs_bounded2_auto.txt")
        self.m = self.theta[0:3*(self.Nm-1)].reshape(self.Nm-1,3) #array of m-nodes
        self.b = self.theta[3*(self.Nm-1):3*(self.Nm+self.Nb-2)].reshape(self.Nb-1,3) #array of b-nodes
        self.lnf = self.theta[3*(self.Nm+self.Nb-2):].reshape(self.Nf-1,3) #array of lnf-nodes

        ####################################HACKY ###############################
        self.bnod = self.bnod[:-1]
        self.mnod = self.mnod[:-1]
        self.fnod = self.fnod[:-1]
        self.m = self.m[:-1,:]
        self.b = self.b[:-1,:]
        self.lnf = self.lnf[:-1,:]
        ####################################################################
        red_file = h5py.File("red_cat_auto.hdf5" , 'r')
        red_sample = red_file['red'][:]
        mrefs = red_file['mref'][:]
        red_file.close()
        znods = np.linspace(self.zmin, self.zmax, 36)
	znods = .5*(znods[1:]+znods[:-1])
	#####################################HACKY#############################
        mrefs = mrefs[znods<0.7]
        znods = znods[znods<0.7]
	#######################################################################
        self.xref = CubicSpline(znods, mrefs)(self.xrefnod)

        return None

    def interpolate(self , z):
        """
        interpolate a , b , c arrays from 
        nodes to a given z
        """

        bz = CubicSpline(self.bnod , self.b)(z)
        mz = CubicSpline(self.mnod , self.m)(z)
        lnfz = CubicSpline(self.fnod , self.lnf)(z)
        xrefz = CubicSpline(self.xrefnod , self.xref)(z)
        
        return mz, bz, lnfz, xrefz
   
    def lnredsq(self, z ,mag ,color ,cerr):
        
        lp = lnprior(z, self.zmin, self.zmax)
    	if not np.isfinite(lp):
		return -np.inf
               
	mz, bz, lnfz, xrefz = self.interpolate(z)
	mz, bz, lnfz = mz[0], bz[0], np.exp(2. * lnfz[0])
        cov_tot = cerr + np.diag([lnfz[0], lnfz[1], lnfz[2]]) 
        cmod = mz * (mag - xrefz) + bz
        dc = color - cmod
        lnred = -0.5 * np.dot(dc.T, np.linalg.solve(cov_tot, dc)) - 0.5 * np.log(np.linalg.det(cov_tot))
        
        return lnred + schecter(mag,z) + lp[0]

    def lnredline(self, z ,mag ,color ,cerr):
        
        lp = lnprior(z, self.zmin, self.zmax)
    	if not np.isfinite(lp):
		return -np.inf
               
	mz, bz, lnfz, xrefz = self.interpolate(z)
	mz, bz, lnfz = mz[0], bz[0], np.exp(2. * lnfz[0])
        cov_tot = cerr + np.diag([lnfz[0], lnfz[1], lnfz[2]]) 
        cmod = mz * (mag - xrefz) + bz
        dc = color - cmod
        lnred = np.dot(dc.T, np.linalg.solve(cov_tot, dc))
        
        return lnred


def action(argz):
    
    imin , imax = argz

    estimator = estimate(zmin , zmax , dz, 15, 8, 6)
    lnpost = estimator.lnredsq
    lnredln = estimator.lnredline
    
    #sample_file = h5py.File("dense_errors.hdf5" , 'r')
    temp_zerr = [] 
    for i in range(imin, imax):

        	def lnprob(p):
            		return lnpost(p, mi_dense[i], colors_dense[i] , cerrs_dense[i])
       
                ndim, nwalkers = 1, 10
		pos = [z_lum[i] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=())
                sampler.run_mcmc(pos, 120)
                samples = sampler.chain[:, 60:, :].reshape((-1, ndim))
		z_percentiles = np.percentile(samples, [16, 84]) 
		zerr = 0.5 * (z_percentiles[1] - z_percentiles[0])
		temp_zerr.append(zerr)
    
    return np.array(temp_zerr)


def sampler(zmin, zmax , dz , nwalkers , nburn, npro, Nthreads):

    print "begin sampling"
    
    import multiprocessing
    from multiprocessing import Pool
   
    pool = Pool(Nthreads)
    mapfn = pool.map
    arglist = [None] * Nthreads
    galnods = np.split(np.arange(len(z_dense)) , Nthreads) 
    
    for j in range(Nthreads):
         arglist[j] = (galnods[j][0] , galnods[j][-1])
         print arglist[j]
    #for ars in arglist:
    #    action(ars)
    #print arglist   
    result2 = list(mapfn(action, [ars for ars in arglist]))
    print result2
    result2 = np.array(result2).flatten()
    
    np.savetxt("dense_err.txt" , result2)  

    return None

if __name__ == '__main__':

   from scipy.stats import norm
   import scipy.optimize as op
   
   lrg_dense = h5py.File("LRG_lmin_0.5_nbar_0.001_auto.h5" , "r")
   lrg_lum = h5py.File("LRG_lmin_1.0_nbar_0.0002_auto.h5" , "r")
   z_dense = lrg_dense["redshift"][:]
   ID_dense = lrg_dense["ID"][:]
   mi_dense = lrg_dense["mi"][:]
   colors_dense = lrg_dense["colors"][:]
   cerrs_dense = lrg_dense["color_errs"][:]

   z_lum = lrg_lum["redshift"][:]
   ID_lum = lrg_lum["ID"][:]
   mi_lum = lrg_lum["mi"][:]
   colors_lum = lrg_lum["colors"][:]
   cerrs_lum = lrg_lum["color_errs"][:]


   model = ezgal.model("data/bc03_burst_0.1_z_0.02_salp.model")
   zf = 3.0 #HARDCODED
   model.add_filter("data/i.dat" , "kids" , units = "nm")
   #kcorr_sloan = model.get_kcorrects(zf=3.0 , zs = 0.25 , filters = "sloan_i")
   model.set_normalization("sloan_i" , 0.2 , 17.85, vega=False, apparent=True)
    
   cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':1.0}
   zmin , zmax , dz = 0.1 , 0.8 , 0.02
   sampler(0.1, 0.8, 0.02, 10, 1000, 2000, 54)
