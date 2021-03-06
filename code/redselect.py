'''
class for estimating
p(z|m,c) \propto p(c|m,z)p(m|z)p(z)
'''
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import CubicSpline
import ezgal
import cosmolopy.distance as cd
import util
import kids_gama
import emcee
from mixture_filtering import catalog_combinator

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
    if z < 0.5 :
       return 22.44 + 3.36*np.log(z) + 0.273*np.log(z)**2 - 0.0618*np.log(z)**3 - 0.0227*np.log(z)**4
    elif z > 0.5 :
       return 22.94 + 3.08*np.log(z) - 11.22*np.log(z)**2 - 27.11*np.log(z)**3 -18.02*np.log(z)**4

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

        self.theta = np.loadtxt("opt_theta.txt")
        self.m = self.theta[0:3*(self.Nm-1)].reshape(self.Nm-1,3) #array of m-nodes
        self.b = self.theta[3*(self.Nm-1):3*(self.Nm+self.Nb-2)].reshape(self.Nb-1,3) #array of b-nodes
        self.lnf = self.theta[3*(self.Nm+self.Nb-2):].reshape(self.Nf-1,3) #array of lnf-nodes

        red_file = h5py.File("red_cat.hdf5" , 'r')
        red_sample = red_file['red'][:]
        mrefs = red_file['mref'][:]
        red_file.close()
        znods = np.linspace(self.zmin, self.zmax, 36)
        self.xref = CubicSpline(.5*(znods[1:]+znods[:-1]), mrefs)(self.xrefnod)

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
        lnred = -0.5 * np.dot(dc.T, np.linalg.solve(cov_tot, dc)) -0.5 * np.log(np.linalg.det(cov_tot))
        
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
        

def reduce_catalog():

    matched_gals = catalog_combinator().T
    colors = matched_gals[:,8:11] 
    mag_errs = matched_gals[:,4:8]
    color_errs = np.zeros((colors.shape[0], colors.shape[1], colors.shape[1]))
    color_errs[:,0,0] = mag_errs[:,0]**2 + mag_errs[:,1]**2
    color_errs[:,1,1] = mag_errs[:,1]**2 + mag_errs[:,2]**2
    color_errs[:,2,2] = mag_errs[:,2]**2 + mag_errs[:,3]**2
    color_errs[:,0,1] = -1. * mag_errs[:,1]**2
    color_errs[:,1,0] = -1. * mag_errs[:,1]**2
    color_errs[:,1,2] = -1. * mag_errs[:,2]**2
    color_errs[:,2,1] = -1. * mag_errs[:,2]**2
    mi = matched_gals[:,3]
    redshift = matched_gals[:,11]

    return mi, colors, color_errs, redshift

def sampler(zmin, zmax , dz , nwalkers , nburn, npro):

    mi , colors , color_errs, redshift = reduce_catalog() 
    
    Ngals = mi.shape[0]
    
    estimator = estimate(zmin , zmax , dz, 15, 8, 6)
    lnpost = estimator.lnredsq
    lnredln = estimator.lnredline

    ndim, nwalkers = 1, nwalkers
    bounds = [(zmin , zmax)]
    p0 = .5 * (zmax + zmin)
    p0 = [p0 + 1e-2 * np.random.randn(ndim) for k in range(nwalkers)]
    
    result_file = h5py.File("red_photo_v1.h5" , 'w')
    ##result_file.create_dataset("opt", (Ngals, 4), data = np.zeros((Ngals,4)))
    result_file.create_dataset("opt", (Ngals, 3), data = np.zeros((Ngals,3)))
    result_file.close()

   
    for i in range(150000, Ngals):

        def lnprob(p):
            return lnpost(p, mi[i], colors[i] , color_errs[i])
       
        nll = lambda *args: -lnprob(*args)
        result = op.minimize(nll, 0.5)
        chi_red = lnredln(result["x"], mi[i], colors[i] , color_errs[i])
        
        print i
        print "status" , result["success"]
        print "estimate" , result["x"][0] 
	print "truth" , redshift[i]
        print "chi_red" , chi_red 
       
        status = 0.0
        if result["success"] == True : status = 1.0
       
        #lratio = luminosity(mags[i],zs[i])

        sample_file = h5py.File("red_photo_v1.h5")
        #sample_file["opt"][i] = np.array([status , result["x"][0] , zs[i] , chi_red])
        sample_file["opt"][i] = np.array([status , result["x"][0] , chi_red])
        sample_file.close()

        """
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args = (mags[i] , colors[i] , color_errs[i]))
        pos , _ , _ = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, npro)
        chain = sampler.chain
        print np.mean(chain)
        print np.std(chain)
        fig_dir = util.fig_dir()
        sns.distplot(chain.flatten(), fit=norm, kde=False)
        plt.axvline(x = zs[i])
        plt.xlim([0,1])
        plt.savefig(fig_dir+str(i)+".png")
        plt.close()
        """

if __name__ == '__main__':

   from scipy.stats import norm
   import seaborn as sns
   import scipy.optimize as op
   import h5py

   model = ezgal.model("/net/delft/data2/vakili/easy/ezgal_models/www.baryons.org/ezgal/models/bc03_burst_0.1_z_0.02_chab.model")
   model.add_filter("/net/delft/data2/vakili/easy/i.dat" , "kids" , units = "nm")
   kcorr_sloan = model.get_kcorrects(zf=3.0 , zs = 0.2 , filters = "sloan_i")
   model.set_normalization("sloan_i" , 0.2 , redmapper_mstar(0.2)-kcorr_sloan, vega=False, apparent=True)
    
   zf = 3.0 #HARDCODED
   cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72}
   sampler(0.1, 0.8, 0.02, 10, 1000 , 2000)

   #v2 ----> z = 0.6 match , zf = 3 match, but harcoded zf = 5, no log(var)
   #v3 ----> z = 0.25      ,      3      ,                   3, no log(var)
   #v4 ----> z = 0.2 match same as v3 but with variance!
