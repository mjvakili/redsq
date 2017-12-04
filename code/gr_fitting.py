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

def catalog_combinator():
    '''
    combines sdss and gama catalogs 
    '''
    combined_cat = np.hstack([gama_reduction() , sdss_reduction()])

    return combined_cat


def catalog_slicer(zmin, zmax, component):

     
    combined_cat = catalog_combinator()
    z = combined_cat[11,:]
    mask = (z>zmin) & (z<zmax)
    reduced_cat = combined_cat[:,mask]
    

    color = reduced_cat[8:11,:]
    color_err = np.zeros_like(color)
    color_err[0,:] = reduced_cat[4,:]**2. + reduced_cat[5,:]**2.
    color_err[1,:] = reduced_cat[5,:]**2. + reduced_cat[6,:]**2.
    color_err[2,:] = reduced_cat[6,:]**2. + reduced_cat[7,:]**2.

    x = reduced_cat[3,:] #mi the reference magnitude
    y = color[component, :]
    yerr = color[component , :]

    return x, y, yerr

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


def lnlike_fg_onemix(p):

    '''
    likelihood of red galaxies when there are no outliers
    '''
    #m_ref = np.median   
    m, b, lnf= p
    model = m * (x - 19.0) + b
    var = np.exp(lnf) + yerr ** 2
    
    return -0.5 * ((model - y) **2 / var + np.log(var))


# Full probabilistic model.


def lnprob_fg(p):
    '''
    posterior prob of red galaxies only
    '''
    m, b, lnf = p
    # First check the prior.
    lp = lnprior_fg(p)
    if not np.isfinite(lp):
	return -np.inf, None
    # Compute the vector of foreground likelihood.
    ll_fg = lnlike_fg_onemix(p)
    return ll_fg

def lnprior(p):
    
    bounds = [(-0.1, 0.1), (1.0,2.0), (-20.0, -2.0), (0, 1), (0.0, 2.0), (-7.0,5.2)]
    # We'll just put reasonable uniform priors on all the parameters.
    if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
        return -np.inf
    return 0

def lnprior_fg(p):
    
    bounds = [(-0.2, 0.2), (0.1,2.0), (-20.0, -2.0)]
    # We'll just put reasonable uniform priors on all the parameters.
    if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
        return -np.inf
    return 0

def mcmc(args): 

    zmin , zmax , iteration , x, y, yerr = args

    def lnlike_fg(p):

    	'''
    	likelihood of red galaxies in the presence of outliers
    	'''
    	#m_ref = np.median   
    	m, b, lnf, _, _,_ = p
    	model = m * (x - 19.0) + b
    	var = np.exp(lnf) + yerr ** 2
    	return -0.5 * ((model - y) **2 / var + np.log(var))

    def lnlike_bg(p):
    	_, _, _, Q, M, lnV = p
   	var = np.exp(lnV) + yerr**2
   	return -0.5 * ((M - y) ** 2 / var + np.log(var))

    
    def lnprob(p):
    	'''
    	posterior prob when there are outliers
    	has two mixtures : reds and blues
    	'''
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


    ndim, nwalkers = 6, 32
    #bounds = [(-0.2, 0.2), (0.2,1.0), (-20.0, -1.5), (0, 1), (0.0, 1.0), (-8.0,1.5)]
    bounds = [(-0.1, 0.1), (1.0,2.0), (-20.0, -2.0), (0, 1), (0.0, 2.0), (-7.0,5.2)]
    p0 = np.array([0.0, 1.2, np.log(0.01) , 0.7, 1.0, np.log(2.0)])
    #p0 = np.array([0.0, 0.5, np.log(0.01) , 0.7, 0.4, np.log(0.16)])
    p0 = [p0 + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)]
           
    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    # Run a burn-in chain and save the final location.
    pos, _, _, _ = sampler.run_mcmc(p0, 3000)

    from matplotlib.ticker import MaxNLocator
    
    sample = sampler.chain
    npars = sample.shape[2]
    fig , axes = plt.subplots(npars , 1 , sharex=True, figsize=(10, 12))

    for i in xrange(npars):
        axes[i].plot(sample[:, :, i].T, color="b", alpha=.4 , lw = .5)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
	axes[i].set_ylim([bounds[i][0], bounds[i][1]])
	axes[i].set_xlim(0, 5000)
        #axes[i].set_ylabel(labels[i], fontsize=25)
    axes[-1].set_xlabel("Step Number", fontsize=25)
    fig.tight_layout(h_pad=0.0)
    fig_file = "/home/vakili/public_html/files/kidsxsdss_gama/gr/"+str(zmin)+"_z_"+str(zmax)+"burn_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()

    # Run the production chain.
    sampler.reset()
    sampler.run_mcmc(pos, 1000)
    
    print sampler.chain.shape

    import corner
    labels = ["$m$", "$b$", "\ln f", "$Q$", "$M$", "$\ln V$"]
    #truths = true_params + [true_frac, true_outliers[0], np.log(true_outliers[1])]
    #bounds = [(-0.2, 0.2), (0.2,1.0), (-20.0, -1.5), (0, 1), (0.0, 1.0), (-8.0,1.5)]
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
    fig_file = "/home/vakili/public_html/files/kidsxsdss_gama/gr/"+str(zmin)+"_z_"+str(zmax)+"chain_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()
    

    est = np.median(sampler.flatchain , axis = 0)
    est[2] = np.median(np.exp(sampler.flatchain)**.5 , axis = 0)[2] 

    est_err = np.std(sampler.flatchain , axis = 0)
    est_err[2] = np.std(np.exp(sampler.flatchain)**.5 , axis = 0)[2]

    xx = np.linspace(14.5 , 25.5 , 1000)
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
   
    for iii in range(1,2):
	pl = plt.scatter(x, y,
	            c = post_prob , s = 0.3 , cmap = 'viridis' , label = str(zmin)+'<z<'+str(zmax))
        plt.ylabel(labels[iii])
	#axs[0].set_yscale('log')
        plt.xlim([14.5, 24.5])
    plt.text(16 , 0.75, r"$slope = $"+str(round(est[0],4))+"$\pm$"+str(round(est_err[0],5)))	
    plt.text(16 , 0.6 , r"$intercept = $"+str(round(est[1],4))+"$\pm$"+str(round(est_err[1],5)))	
    plt.text(16 , 0.45 , r"$scatter = $"+str(round(est[2],4))+"$\pm$"+str(round(est_err[2],5)))	

    cb = plt.colorbar(pl) 
    plt.legend(loc='best')
    cb.set_label('Red-sequence Membership Probability')
    plt.plot(xx, pred, color="k", lw=1.5)	
    plt.ylim([-0.1, 2.1])
    #axs.set_ylim([-0.1, 1.1])
    plt.xlabel(r'$m_{i}$')
     
    #fig.tight_layout()
    fig_file = "/home/vakili/public_html/files/kidsxsdss_gama/gr/"+str(zmin)+"_z_"+str(zmax)+"color_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()

    return np.concatenate([est, est_err, post_prob])

def mcmc_fg(args): 

    zmin , zmax , iteration , x, y, yerr = args

    def lnlike_fg(p):

    	'''
    	likelihood of red galaxies in the presence of outliers
    	'''
    	#m_ref = np.median   
    	m, b, lnf, _, _,_ = p
    	model = m * (x - 19.0) + b
    	var = np.exp(lnf) + yerr ** 2
    	return -0.5 * ((model - y) **2 / var + np.log(var))

    def lnlike_bg(p):
    	_, _, _, Q, M, lnV = p
   	var = np.exp(lnV) + yerr**2
   	return -0.5 * ((M - y) ** 2 / var + np.log(var))

    
    def lnprob(p):
    	'''
    	posterior prob when there are outliers
    	has two mixtures : reds and blues
    	'''
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


    ndim, nwalkers = 6, 32
    #bounds = [(-0.2, 0.2), (0.2,1.0), (-20.0, -1.5), (0, 1), (0.0, 1.0), (-8.0,1.5)]
    bounds = [(-0.2, 0.2), (1.0,2.0), (-20.0, -2.0), (0, 1), (0.0, 2.0), (-7.0,5.2)]
    p0 = np.array([0.0, 1.2, np.log(0.01) , 0.7, 1.0, np.log(2.0)])
    #p0 = np.array([0.0, 0.5, np.log(0.01) , 0.7, 0.4, np.log(0.16)])
    p0 = [p0 + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)]
           
    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    # Run a burn-in chain and save the final location.
    pos, _, _, _ = sampler.run_mcmc(p0, 3000)

    from matplotlib.ticker import MaxNLocator
    
    sample = sampler.chain
    npars = sample.shape[2]
    fig , axes = plt.subplots(npars , 1 , sharex=True, figsize=(10, 12))

    for i in xrange(npars):
        axes[i].plot(sample[:, :, i].T, color="b", alpha=.4 , lw = .5)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
	axes[i].set_ylim([bounds[i][0], bounds[i][1]])
	axes[i].set_xlim(0, 5000)
        #axes[i].set_ylabel(labels[i], fontsize=25)
    axes[-1].set_xlabel("Step Number", fontsize=25)
    fig.tight_layout(h_pad=0.0)
    fig_file = "/home/vakili/public_html/files/kidsxsdss_gama/gr/"+str(zmin)+"_z_"+str(zmax)+"burn_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()

    # Run the production chain.
    sampler.reset()
    sampler.run_mcmc(pos, 1000)
    
    print sampler.chain.shape

    import corner
    labels = ["$m$", "$b$", "\ln f", "$Q$", "$M$", "$\ln V$"]
    #truths = true_params + [true_frac, true_outliers[0], np.log(true_outliers[1])]
    #bounds = [(-0.2, 0.2), (0.2,1.0), (-20.0, -1.5), (0, 1), (0.0, 1.0), (-8.0,1.5)]
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
    fig_file = "/home/vakili/public_html/files/kidsxsdss_gama/gr/"+str(zmin)+"_z_"+str(zmax)+"chain_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()
    

    est = np.median(sampler.flatchain , axis = 0)
    est[2] = np.median(np.exp(sampler.flatchain)**.5 , axis = 0)[2] 

    est_err = np.std(sampler.flatchain , axis = 0)
    est_err[2] = np.std(np.exp(sampler.flatchain)**.5 , axis = 0)[2]

    xx = np.linspace(14.5 , 25.5 , 1000)
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
   
    for iii in range(1,2):
        pl = plt.errorbar(x, y, yerr=yerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
	#pl = plt.errorbar(x, y, yerr, marker = 'o', color = 'green' , label = str(zmin)+'<z<'+str(zmax))
        plt.ylabel(labels[iii])
	#axs[0].set_yscale('log')
        plt.xlim([14.5, 24.5])
    
    plt.text(16 , 0.75, r"$slope = $"+str(round(est[0],4))+"$\pm$"+str(round(est_err[0],5)))	
    plt.text(16 , 0.6 , r"$intercept = $"+str(round(est[1],4))+"$\pm$"+str(round(est_err[1],5)))	
    plt.text(16 , 0.45 , r"$scatter = $"+str(round(est[2],4))+"$\pm$"+str(round(est_err[2],5)))	

    #cb = plt.colorbar(pl) 
    #cb.set_label('Red-sequence Membership Probability')
    plt.plot(xx, pred, color="#8d44ad", lw=1.5, label = str(zmin)+'<z<'+str(zmax))	
    plt.fill_between(xx, pred-est[2], pred+est[2], color="#8d44ad", alpha=0.1)

    plt.legend(loc='best')
    
    plt.ylim([-0.1, 2.1])
    #axs.set_ylim([-0.1, 1.1])
    plt.xlabel(r'$m_{i}$')
     
    #fig.tight_layout()
    ##fig_file = "/home/vakili/public_html/files/redsequence_gr/"+str(zmin)+"_z_"+str(zmax)+"color_iter"+str(iteration)+".png"
    fig_file = "/home/vakili/public_html/files/kidsxsdss_gama/gr/"+str(zmin)+"_z_"+str(zmax)+"color_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()

    return np.concatenate([est, est_err, post_prob])

if __name__ == '__main__':

   Niter = 5 #number of iterations
   z_init = 0.2
   Nthreads = 1

   import multiprocessing
   from multiprocessing import Pool
   
   pool = Pool(Nthreads)
   mapfn = pool.map
   arglist = [None] * Nthreads
   for i in range(Nthreads):

       zmin = z_init + i*0.02
       zmax = zmin + 0.02
       x, y, yerr = catalog_slicer(zmin , zmax , 1)
       
       ##filename = "results3/gr_result_z_"+str(zmin)+"_"+str(zmax)+".txt"
       ##pmem = np.loadtxt(filename)[12:]
       ##x , y , yerr = x[pmem>0.8] , y[pmem>0.8] , yerr[pmem>0.8]
       
       arglist[i] = (zmin, zmax, 1, x, y, yerr)  
   result = list(mapfn(mcmc, [ars for ars in arglist]))
   ##result = list(mapfn(mcmc_fg, [ars for ars in arglist]))
   
   #for t in range(Nthreads):
   #    zmin = z_init + t * 0.02
   #    zmax = zmin + 0.02
   #    np.savetxt("results4/gr_result_z_"+str(zmin)+"_"+str(zmax)+".txt" , np.array(result[t]))
   pool.close()

   # Now let's use the pmems to apply cuts to the data and fit again.
   """
   for i in range(Nthreads):
       zmin = z_init + i*0.01
       zmax = zmin + 0.01
       x, y, yerr = test(zmin , zmax , 2)
       arglist[i] = (zmin, zmax , 8, x, y , yerr)  
   result = list(mapfn(mcmc, [ars for ars in arglist]))
   
   for t in range(Nthreads):
       zmin = z_init + t * 0.01
       zmax = zmin + 0.01
       np.savetxt("results/ri_result_z_"+str(zmin)+"_"+str(zmax)+".txt" , np.array(result[t]))
   pool.close()
   """
   """
   for i in range(16 , 17):
     
     zmin = z_init + i * 0.02
     zmax = zmin + 0.02
     x, y, yerr = test(zmin , zmax , 2)
	  
     red_prob = mcmc(zmin , zmax , 4)

     ####  removing the outliers p<0.9 ####
     x = x[red_prob > 0.8]
     y = y[red_prob > 0.8]
     yerr = yerr[red_prob > 0.8]
      
     red_prob = mcmc(zmin , zmax , 5) 
    
     ####  removing the outliers p<0.9 ####
     #x = x[red_prob > 0.9]
     #y = y[red_prob > 0.9]
     #yerr = yerr[red_prob > 0.9]
      
     #mcmc_fg(zmin , zmax , 2) 
     #### fitting again after removing the outliers ####
     #red_prob = mcmc(zmin , zmax , 1)
   """ 
