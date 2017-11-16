'''
class for estimating
p(z|m,c) \propto p(c|m,z)p(m|z)p(z)

p(c|m,z) constructed by spline 
interpolation of the best-fit 
c-m relation parameters


p(m|z) is given by the 
schecter fuction with m_char
derived from BC03, KIDS 
bandpass, and EZGAL fella

p(z) = dV_{comoving}/dz
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

    def __init__(self, zmin, zmax, dz):
 
        self.zmin = zmin #minimum redshift
        self.zmax = zmax #maximum redshift
        self.dz = dz     #width of redshift bins

        cm_dir = util.cm_dir()
        Nbin = np.int((self.zmax - self.zmin) / self.dz)
        cov , ug , gr , ri = [] , [] , [] , []
        cm_dir = util.cm_dir()

        for i in xrange(Nbin):

            z1 = self.zmin + self.dz * i
            z2 = self.zmin + self.dz * (i+1)
            file_ug = cm_dir+"ug_result_z_"+str(z1)+"_"+str(z2)+".txt"     
            file_gr = cm_dir+"gr_result_z_"+str(z1)+"_"+str(z2)+".txt"   
            file_ri = cm_dir+"ri_result_z_"+str(z1)+"_"+str(z2)+".txt"     
            file_cov = cm_dir+"covariance_z_"+str(z1)+"_"+str(z2)+".txt"     
            ug.append(np.loadtxt(file_ug)[:2])
            gr.append(np.loadtxt(file_gr)[:2])
            ri.append(np.loadtxt(file_ri)[:2])
            cov.append(np.loadtxt(file_cov))

        self.cov , self.ug , self.gr , self.ri = np.array(cov) , np.array(ug) , \
                                                 np.array(gr) , np.array(ri)
        return None

    def interpolate(self , z):
        """
        interpolate a , b , c arrays from 
        nodes to a given z
        """
        Nbin = np.int((self.zmax - self.zmin)/self.dz)
        z_input = np.linspace(self.zmin , self.zmax , Nbin)
        cov_z = CubicSpline(z_input, self.cov)(z)
        ug_z = CubicSpline(z_input, self.ug)(z)
        gr_z = CubicSpline(z_input, self.gr)(z)
        ri_z = CubicSpline(z_input, self.ri)(z)
        
        return cov_z[0] , ug_z[0] , gr_z[0] , ri_z[0]
   
    def lnredsq(self, z ,mag ,c ,cerr):
        
        lp = lnprior(z, self.zmin, self.zmax)
    	if not np.isfinite(lp):
		return -np.inf
                
        cov_int_z, ug_z, gr_z , ri_z = self.interpolate(z)
        ue, ge, re, ie = cerr
        cov_obs = np.array([[ue**2+ge**2, -1.*ge**2, 0.], [-1.*ge**2, ge**2+re**2, -1.*re**2], [0., -1.*re**2, re**2.+ie**2.]])
        cov_tot = cov_int_z + cov_obs
        

        slope  = np.array([ug_z[0],gr_z[0],ri_z[0]])
        incpt  = np.array([ug_z[1],gr_z[1],ri_z[1]])

        cmod = slope * (mag - 19) + incpt
        dc = c - cmod
        lnred = -0.5 * np.dot(dc.T, np.linalg.solve(cov_tot, dc))
        
        return lnred + schecter(mag,z) + lp[0]

    def lnredline(self, z ,mag ,c ,cerr):
        
                
        cov_int_z, ug_z, gr_z , ri_z = self.interpolate(z)
        ue, ge, re, ie = cerr
        cov_obs = np.array([[ue**2+ge**2, -1.*ge**2, 0.], [-1.*ge**2, ge**2+re**2, -1.*re**2], [0., -1.*re**2, re**2.+ie**2.]])
        cov_tot = cov_int_z + cov_obs
        

        slope  = np.array([ug_z[0],gr_z[0],ri_z[0]])
        incpt  = np.array([ug_z[1],gr_z[1],ri_z[1]])

        cmod = slope * (mag - 19) + incpt
        dc = c - cmod
        lnred = np.dot(dc.T, np.linalg.solve(cov_tot, dc))

        return lnred

def reduce_catalog(zmin, zmax):

    matched_gals = kids_gama.filter('GAMA-MATCHED')
    
    ug = matched_gals['COLOR_GAAPHOM_U_G'] 
    gr = matched_gals['COLOR_GAAPHOM_G_R'] 
    ri = matched_gals['COLOR_GAAPHOM_R_I']
    z = matched_gals['Z']
    mi = matched_gals['MAG_GAAP_i_CALIB']
    colors = np.array([ug , gr , ri])
    u_err = matched_gals['MAGERR_GAAP_U']
    g_err = matched_gals['MAGERR_GAAP_G']
    i_err = matched_gals['MAGERR_GAAP_I']
    r_err = matched_gals['MAGERR_GAAP_R']
    c_err = np.array([u_err, g_err, r_err, i_err])
    mi = mi[(zmin<z)&(z<zmax)]
    redshift_mask = np.where((zmin<z)&(z<zmax))[0]
    colors = colors[:, redshift_mask]
    c_err = c_err[: , redshift_mask]
    x = mi
    y = colors.T
    yerr = c_err.T

    return x, y, yerr, z[redshift_mask]

def sampler(zmin, zmax , dz , nwalkers , nburn, npro):

    mags , colors , color_errs , zs = reduce_catalog(zmin , zmax) 
    Ngals = mags.shape[0]
    
    estimator = estimate(zmin , zmax , dz)
    lnpost = estimator.lnredsq
    lnredln = estimator.lnredline

    ndim, nwalkers = 1, nwalkers
    bounds = [(zmin , zmax)]
    p0 = .5 * (zmax + zmin)
    p0 = [p0 + 1e-2 * np.random.randn(ndim) for k in range(nwalkers)]
    
    result_file = h5py.File("result.h5" , 'w')
    result_file.create_dataset("opt", (Ngals, 4), data = np.zeros((Ngals,4)))
    result_file.close()

    for i in xrange(Ngals):

        def lnprob(p):
            return lnpost(p, mags[i], colors[i] , color_errs[i])
       
        nll = lambda *args: -lnprob(*args)
        result = op.minimize(nll, 0.3)
        chi_red = lnredln(result["x"] , mags[i], colors[i] , color_errs[i])
        
        print i
        print "status" ,result["success"]
        print "estimate" , result["x"][0] 
        print "truth" , zs[i]
        print "chi_red" , chi_red 
       
        status = 0.0
        if result["success"] == True : status = 1.0
        
        sample_file = h5py.File("result.h5")
        sample_file["opt"][i] = np.array([status , result["x"][0] , zs[i] , chi_red])
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
   kcorr_sloan = model.get_kcorrects(zf=3.0 , zs = 0.25 , filters = "sloan_i")
   model.set_normalization("sloan_i" , 0.25 , redmapper_mstar(0.25)-kcorr_sloan, vega=False, apparent=True)
    
   zf = 3.0 #HARDCODED
   cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.67}
   sampler(0.2, 0.4, 0.02, 10, 1000 , 2000)
