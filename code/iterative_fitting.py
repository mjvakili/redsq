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

def test(zmin , zmax , component):
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
    colors = np.array([ug , gr , ri])
    
    u_err = matched_gals['MAGERR_GAAP_U']
    g_err = matched_gals['MAGERR_GAAP_G']
    i_err = matched_gals['MAGERR_GAAP_I']
    r_err = matched_gals['MAGERR_GAAP_R']
    
    c_err = np.array([u_err**2. + g_err**2.,
                      g_err**2. + r_err**2.,
		      r_err**2. + i_err**2.])**0.5 

    mi = mi[(zmin<z)&(z<zmax)]
    redshift_mask = np.where((zmin<z)&(z<zmax))[0]
    colors = colors[:, redshift_mask]
    c_err = c_err[: , redshift_mask]
  
    x = mi
    y = colors[component,:]
    yerr = c_err[component,:]
    
    return x , y , yerr

def lnprior(p):
    
    bounds = [(-0.9, 0.9), (0.0,2.0), (-5.0, 0), (0, 1), (0.0, 2.0), (-7.2, 5.2)]
    # We'll just put reasonable uniform priors on all the parameters.
    if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
        return -np.inf
    return 0

def lnlike_fg(p):

    '''
    m_ref is the reference magnitude at the redshift bin 
    z : [zmin , zmax]
    m_ref is set to be the median i-band (apparent) 
    magnitude of the BCG's between zmin and zmax
    '''
    #m_ref = np.median   
    m, b, lnf, _, M, lnV = p
    model = m * (x - 18.0) + b
    var = np.exp(lnf) + yerr ** 2
    return -0.5 * ((model - y) **2 / var + np.log(var))

def lnlike_bg(p):
    _, _, _, Q, M, lnV = p
    var = np.exp(lnV) + yerr**2
    return -0.5 * ((M - y) ** 2 / var + np.log(var))

# Full probabilistic model.
def lnprob(p):
    m, b, lnf, Q, M, lnV = p
    # First check the prior.
    lp = lnprior(p)
    if not np.isfinite(lp):
	return -np.inf, None
   
    # Compute the vector of foreground likelihoods and include the q prior.
    ll_fg = lnlike_fg(p)
    arg1 = ll_fg + np.log(Q)

    # Compute the vector of background likelihoods and include the q prior.
    ll_bg = lnlike_bg(p)
    arg2 = ll_bg + np.log(1.0 - Q)

    # Combine these using log-add-exp for numerical stability.
    ll = np.sum(np.logaddexp(arg1, arg2))

    # We're using emcee's "blobs" feature in order to keep track of the
    # foreground and background likelihoods for reasons that will become
    # clear soon.
    return lp + ll, (arg1, arg2)

def mcmc(zmin , zmax):     

    ndim, nwalkers = 6, 32
    bounds = [(-0.2, 0.2), (0.5,1.5), (-20.0, 0), (0, 1), (0.0, 2.0), (-7.2, 5.2)]
    p0 = np.array([0.0, 1.0, np.log(0.1) , 0.7, 1.0, np.log(2.0)])
    p0 = [p0 + 1e-3 * np.random.randn(ndim) for k in range(nwalkers)]
 
    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    # Run a burn-in chain and save the final location.
    pos, _, _, _ = sampler.run_mcmc(p0, 700)


    # Run the production chain.
    sampler.reset()
    sampler.run_mcmc(pos, 500)
    
    print sampler.chain.shape

    import corner
    labels = ["$m$", "$b$", "\ln f", "$Q$", "$M$", "$\ln V$"]
    #truths = true_params + [true_frac, true_outliers[0], np.log(true_outliers[1])]
    bounds = [(-0.2, 0.2), (0.5,1.5), (-20.0, 0), (0, 1), (0.0, 2.0), (-7.2, 5.2)]
    #corner.corner(sampler.flatchain, bins=35, range=bounds, labels=labels)
 
    #plt.savefig("/home/vakili/public_html/files/mcmc.png")
    #plt.close()
    sample = sampler.chain
    npars = sample.shape[2]
    fig , axes = plt.subplots(npars , 1 , sharex=True, figsize=(10, 12))

    from matplotlib.ticker import MaxNLocator

    for i in xrange(npars):
        axes[i].plot(sample[:, :, i].T, color="b", alpha=.4 , lw = .5)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
	axes[i].set_ylim([bounds[i][0], bounds[i][1]])
	axes[i].set_xlim(0, 1500)
        #axes[i].set_ylabel(labels[i], fontsize=25)
    axes[-1].set_xlabel("Step Number", fontsize=25)
    fig.tight_layout(h_pad=0.0)
    fig_file = "/home/vakili/public_html/files/mcmc_time_"+str(zmin)+"<z<"+str(zmax)+".png"
    plt.savefig(fig_file)
    plt.close()

    est = np.median(sampler.flatchain , axis = 0)
    xx = np.linspace(16 , 23 , 1000)
    pred = est[1] + est[0]*(xx - 19)

    norm = 0.0
    post_prob = np.zeros(len(x))
    for i in range(sampler.chain.shape[1]):
        for j in range(sampler.chain.shape[0]):
	        ll_fg, ll_bg = sampler.blobs[i][j]
		post_prob += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
	        norm += 1
    post_prob /= norm

    print post_prob
    
    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    # Load and plot the templates and filters
    palette = itertools.cycle(sns.color_palette())
    plt.figure(figsize=(5.5, 6))
   
    for i in range(1):
	pl = plt.scatter(x, y,
	            c = post_prob , s = 0.3 , cmap = 'viridis' , label = str(zmin)+'<z<'+str(zmax))
        plt.ylabel(labels[i+1])
	#axs[0].set_yscale('log')
        plt.xlim([16.5, 21.5])
    cb = plt.colorbar(pl) 
    plt.legend(loc='best')
    cb.set_label('Red-sequence Membership Probability')
    plt.plot(xx, pred, color="k", lw=1.5)	
    plt.ylim([-0.1, 2.1])
    #axs.set_ylim([-0.1, 1.1])
    plt.xlabel(r'$m_{i}$')
     
    #fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/cm_"+str(zmin)+"<z<"+str(zmax)+".png")
    plt.close()

    return post_prob

def plot():

    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    # Load and plot the templates and filters
    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True)
   
    for i in range(2):
	axs[i].errorbar(mi, colors[i+1,:], yerr = c_err[i+1,:],
	                c= next(palette), fmt = 'o')
        axs[i].legend(loc='lower right', ncol=2)
        axs[i].set_ylabel(labels[i+1])
	#axs[0].set_yscale('log')
        axs[i].set_xlim([16.5, 21.5])
    axs[0].set_ylim([-0.1, 2.1])
    axs[1].set_ylim([-0.1, 1.1])
    axs[1].set_xlabel(r'$m_{i}$')
   
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/testerr_groups.png")
    plt.close()
    
    return None


def XD_filter(y , yerr):

    clf = XDGMM(n_components = 2 , n_iter = 4)
    Y = y.reshape(y.shape[0] , 1) 
    Yerr = np.zeros((y.shape[0] , 1, 1))
    #diag = np.arange(Y.shape[-1])
    Yerr[:, 0, 0] = yerr ** 2
    clf.fit(Y , Yerr)

    return clf.mu, clf.V


if __name__ == '__main__':

   Niter = 4  #number of iterations

   z_init = 0.06

   for i in range(10 , 20):
     
     zmin = z_init + i * 0.02
     zmax = zmin + 0.02

     #zmin, zmax = 0.1, 0.12
     x, y, yerr = test(zmin , zmax , 2)
  
     muV = XD_filter(y , yerr)
     mus , Vs = muV[0] , muV[1]
   
     mu_high , mu_low = mus[0][0] , mus[1][0]
     Vs_high , Vs_low = Vs[0][0,0] , Vs[1][0,0]

     Vs_high , Vs_low = max(Vs_high , Vs_low) , min(Vs_high , Vs_low)

     mu_high , mu_low = max(mu_high , mu_low) , min(mu_high , mu_low)
     
     print mu_high , mu_low
     print Vs_high , Vs_low
    
     plt.figure(figsize = (6,6))
     plt.hist(y , normed = True , alpha = 0.2 , bins = 20)
     x = np.linspace(y.min() , y.max(), 1000)
   
     from scipy.stats import norm
     dist1 = norm(mu_high , Vs_low**.5)
     dist2 = norm(mu_low , Vs_high**.5)
     plt.axvline(mu_high, color='r', linestyle='dashed', linewidth=2)
     #plt.axvline(mu_high - Vs_low**.5, color='k', linestyle='dashed', linewidth=2)
     plt.axvline(mu_high - 2. * Vs_low**.5, color='k', linestyle='dashed', linewidth=2)
     plt.axvline(mu_high + 2. * Vs_low**.5, color='k', linestyle='dashed', linewidth=2)
     plt.plot(x, dist1.pdf(x) , "r-" , label = str(zmin)+"<z<"+str(zmax))
     plt.plot(x, dist2.pdf(x) , "b-")
     plt.xlabel(r"$r-i$" , fontsize = 20)
     plt.xlim([0. , 2.5])
     plt.ylabel("normalized counts" , fontsize = 20)
     plt.legend(loc = "best" , fontsize = 10)
     plt.savefig("/home/vakili/public_html/files/redsequence/GMM2_"+str(zmin)+"<z<"+str(zmax)+".png")
     plt.close()


   #for i in range(Niter):

   #    red_prob = mcmc(zmin , zmax)
   #    x = x[red_prob > 0.5]
   #    y = y[red_prob > 0.5]
   #    yerr = yerr[red_prob > 0.5]

