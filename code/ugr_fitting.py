import pyfits as pf
import matplotlib.pyplot as plt
import multiprocessing
import emcee
import numpy as np

plt.switch_backend("Agg")
from mpl_toolkits.axes_grid1 import AxesGrid

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
    y = colors[0:2,:].T
    yerr = c_err[0:2,:].T
    
    return x , y , yerr

def XD_filter(y , yerr):

    clf = XDGMM(n_components = 2 , n_iter = 4)
    Y = y.reshape(y.shape[0] , 1) 
    Yerr = np.zeros((y.shape[0] , 1, 1))
    #diag = np.arange(Y.shape[-1])
    Yerr[:, 0, 0] = yerr ** 2
    clf.fit(Y , Yerr)

    return clf.mu, clf.V


def lnlike_fg(p):

    '''
    likelihood of red galaxies in the presence of outliers
    '''
    m1, m2, b1, b2, lnf1, lnf2 , _, _, _, _, _  = p
    
    model = np.zeros_like((y))
    model[:,0] = m1 * (x - 19.0) + b1
    model[:,1] = m2 * (x - 19.0) + b2
    
    var = np.zeros_like((yerr))
    var[:,0] = np.exp(lnf1) + yerr[:,0] ** 2
    var[:,1] = np.exp(lnf2) + yerr[:,1] ** 2
   
    return -0.5 * ((model - y) **2 / var + np.log(var))


def lnlike_bg(p):
    _, _, _, _, _, _, Q, M1, M2, lnV1, lnV2 = p
    
    model_bg = np.zeros_like((y))
    model_bg[:,0] = M1
    model_bg[:,1] = M2

    var_bg = np.zeros_like((yerr))
    var_bg[:,0] = np.exp(lnV1) + yerr[:,0]**2
    var_bg[:,1] = np.exp(lnV2) + yerr[:,1]**2

    return -0.5 * ((model_bg - y) ** 2 / var_bg + np.log(var_bg))

# Full probabilistic model.
def lnprob(p):
    '''
    posterior prob when there are outliers
    has two mixtures : reds and blues
    '''
    m1, m2, b1, b2, lnf1, lnf2, Q, M1, M2, lnV1, lnV2 = p
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
    
    #bounds = [(-0.5, 0.5), (1.5,3.0), (-20.0, -1.0), (0, 1), (0.0, 2.0), (-7.2, 5.2)]
    #the above from tested u-g
    m1, m2, b1, b2, lnf1, lnf2, Q, M1, M2, lnV1, lnV2= p
    bounds = [(-0.2, 0.2), (-0.2, 0.2),\
              (1.5, 3.0) , (0.7, 2.0) ,\
	      (-20.0,-1.0),(-20.0,-2.0),\
	      (0, 1),(0.0, 2.0), (0, 4.0),\
	      (-7.2, 5.2),(-7.2, 5.2)]
    # We'll just put reasonable uniform priors on all the parameters.
    if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
        return -np.inf
    return 0


def mcmc(zmin , zmax , iteration , nburn , nprod):     


    ndim, nwalkers = 11, 30
    bounds = [(-0.2, 0.2), (-0.2, 0.2),\
              (1.5, 3.0) , (0.7, 2.0) ,\
	      (-20.0,-1.0),(-20.0,-2.0),\
	      (0, 1),(0.0, 2.0), (0, 4.0),\
	      (-7.2, 5.2),(-7.2, 5.2)]
    #p0 = np.array([0.0, 1.7, np.log(0.1) , 0.7, 1.0, np.log(2.0)])
    #the above is from a well-tested ug fit
    p0 = np.array([0.0, 0.0, 1.7, 1.0, np.log(0.1), np.log(0.1),
                   0.7,1.0,0.4,np.log(2.0),np.log(2.0)])
    p0 = [p0 + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)]
           
    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    # Run a burn-in chain and save the final location.
    pos, _, _, _ = sampler.run_mcmc(p0, nburn)

    from matplotlib.ticker import MaxNLocator
    
    sample = sampler.chain
    npars = sample.shape[2]
    fig , axes = plt.subplots(npars , 1 , sharex=True, figsize=(10, 12))
    labels = ["m1" , "m2" , "b1" , "b2" , "lnf1" , "lnf2" , "Q" , "M1" , "M2" , "lnV1" , "lnV2"]
    for i in xrange(npars):
        axes[i].plot(sample[:, :, i].T, color="b", alpha=.4 , lw = .5)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
	axes[i].set_ylim([bounds[i][0], bounds[i][1]])
	axes[i].set_xlim(0, 5000)
        axes[i].set_ylabel(labels[i], fontsize=25)
    axes[-1].set_xlabel("Step Number", fontsize=25)
    fig.tight_layout(h_pad=0.0)
    fig_file = "/home/vakili/public_html/files/redsequence_ugr/"+str(zmin)+"_z_"+str(zmax)+"burn_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()

    # Run the production chain.
    sampler.reset()
    sampler.run_mcmc(pos, nprod)
    
    print sampler.chain.shape

    import corner
    labels = ["m1" , "m2" , "b1" , "b2" , "lnf1" , "lnf2" , "Q" , "M1" , "M2" , "lnV1" , "lnV2"]
    #truths = true_params + [true_frac, true_outliers[0], np.log(true_outliers[1])]
    bounds = [(-0.2, 0.2), (-0.2, 0.2),\
              (1.5, 3.0) , (0.7, 2.0) ,\
	      (-20.0,-1.0),(-20.0,-2.0),\
	      (0, 1),(0.0, 2.0), (0, 4.0),\
	      (-7.2, 5.2),(-7.2, 5.2)]
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
        axes[i].set_ylabel(labels[i], fontsize=25)
    axes[-1].set_xlabel("Step Number", fontsize=25)
    fig.tight_layout(h_pad=0.0)
    fig_file = "/home/vakili/public_html/files/redsequence_ugr/"+str(zmin)+"_z_"+str(zmax)+"chain_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()
    
    

    norm = 0.0
    post_prob = np.zeros(len(x))
    for i in range(sampler.chain.shape[1]):
        for j in range(sampler.chain.shape[0]):
	        ll_fg, ll_bg = sampler.blobs[i][j]
		#print "likelihood shape" , ll_fg.shape , ll_bg.shape
		#post_prob += np.exp(np.sum(ll_fg - np.logaddexp(ll_fg, ll_bg))
		#print "logaddexp shape" , np.logaddexp(ll_fg , ll_bg).shape
		post_prob += np.exp(np.sum(ll_fg - np.logaddexp(ll_fg, ll_bg), axis = 1))
	        norm += 1		
    post_prob /= norm

    print post_prob
    
    """ 
    est = np.median(sampler.flatchain , axis = 0)
    est[2] = np.median(np.exp(sampler.flatchain)**.5 , axis = 0)[2] 

    est_err = np.std(sampler.flatchain , axis = 0)
    est_err[2] = np.std(np.exp(sampler.flatchain)**.5 , axis = 0)[2]

    xx = np.linspace(14.5 , 25.5 , 1000)
    pred = est[1] + est[0]*(xx - 19)
    """
    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    # Load and plot the templates and filters
    from matplotlib import gridspec
    fig = plt.figure(figsize=(12,6))
    gs=gridspec.GridSpec(1,3, width_ratios=[4,4,0.2])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    cm = plt.cm.get_cmap('RdYlBu')
    ax1.scatter(x, y[:,0], s = 0.2 , c=post_prob, cmap=cm)
    SC2 = ax2.scatter(x, y[:,1], s = 0.2 , c=post_prob, cmap=cm)
    plt.colorbar(SC2, cax=ax3)
    plt.tight_layout()
    #plt.setp(ax2.get_yticklabels(), visible=False)
    #plt.tight_layout()
    fig_file = "/home/vakili/public_html/files/redsequence_ugr/unclean_"+str(zmin)+"_z_"+str(zmax)+"color_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()

    from matplotlib import gridspec
    fig = plt.figure(figsize=(12,6))
    gs=gridspec.GridSpec(1,3, width_ratios=[4,4,0.2])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    cm = plt.cm.get_cmap('RdYlBu')
    ax1.scatter(x[post_prob>0.8], y[post_prob>0.8][:,0], s = 0.2 , c=post_prob[post_prob>0.8], cmap=cm)
    SC2 = ax2.scatter(x[post_prob>0.8], y[post_prob>0.8][:,1], s = 0.2 , c=post_prob[post_prob>0.8], cmap=cm)
    plt.colorbar(SC2, cax=ax3)
    plt.tight_layout()
    #plt.setp(ax2.get_yticklabels(), visible=False)
    #plt.tight_layout()
    fig_file = "/home/vakili/public_html/files/redsequence_ugr/clean_"+str(zmin)+"_z_"+str(zmax)+"color_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()
    
    palette = itertools.cycle(sns.color_palette())
    fig = plt.figure(figsize=(6, 6))
    grid = AxesGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.05, cbar_mode='single',cbar_location='right',cbar_pad=0.1)
    										                    
    #fig, axes = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True)
    iii = 0


    for ax in grid:
        #ax.set_axis_off()
	im = ax.scatter(x, y[:,iii], 
	               c = post_prob , s = 0.2 , cmap = 'viridis' , vmin=0, vmax=1, label = str(zmin)+'<z<'+str(zmax))
        ax.set_ylabel(labels[iii])
	iii +=1
        ax.set_xlim([14.5, 24.5])
        cbar = ax.cax.colorbar(im)
    # cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
    cbar.ax.set_yticks(np.arange(0, 1.1, 0.2))
    cbar.ax.set_yticklabels(['0', '0.2','0.4','0.6','0.8','1'])
    #plt.text(16 , 0.75, r"$slope = $"+str(round(est[0],4))+"$\pm$"+str(round(est_err[0],5)))	
    #plt.text(16 , 0.6 , r"$intercept = $"+str(round(est[1],4))+"$\pm$"+str(round(est_err[1],5)))	
    #plt.text(16 , 0.45 , r"$scatter = $"+str(round(est[2],4))+"$\pm$"+str(round(est_err[2],5)))	

    #cb = plt.colorbar() 
    #plt.legend(loc='best')
    #cb.set_label('Red-sequence Membership Probability')
    #plt.plot(xx, pred, color="k", lw=1.5)	
    axes[0].set_ylim([-0.1, 4.1])
    axes[1].set_ylim([-0.1, 2.1])
    axes[1].set_xlabel(r'$m_{i}$')
     
    #fig.tight_layout()
    fig_file = "/home/vakili/public_html/files/redsequence_ugr/"+str(zmin)+"_z_"+str(zmax)+"color_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()
    
    return post_prob


def mcmc_fg(zmin , zmax , iteration):     


    ndim, nwalkers = 3, 32
    #bounds = [(-0.2, 0.2), (0.7,2.0), (-20.0, -2.0)]
 
    bounds = [(-0.5, 0.5), (1.5,3.0), (-20.0, -1.0)]
    p0 = np.array([0.0, 1.0, np.log(0.1)])
    p0 = [p0 + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)]
           
    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_fg)

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
    fig_file = "/home/vakili/public_html/files/redsequence_ug/"+str(zmin)+"_z_"+str(zmax)+"fg_burn_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()

    # Run the production chain.
    sampler.reset()
    sampler.run_mcmc(pos, 1000)
    
    print sampler.chain.shape

    import corner
    labels = ["$m$", "$b$", "\ln f"]
    #truths = true_params + [true_frac, true_outliers[0], np.log(true_outliers[1])]
    bounds = [(-0.2, 0.2), (0.7,2.0), (-20.0, -2.0)]
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
    fig_file = "/home/vakili/public_html/files/redsequence_ug/"+str(zmin)+"_z_"+str(zmax)+"fg_chain_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()

    est = np.median(sampler.flatchain , axis = 0)
    est[2] = np.median(np.exp(sampler.flatchain)**.5 , axis = 0)[2] 

    est_err = np.std(sampler.flatchain , axis = 0)
    est_err[2] = np.std(np.exp(sampler.flatchain)**.5 , axis = 0)[2]

    xx = np.linspace(14.5 , 25.5 , 1000)
    pred = est[1] + est[0]*(xx - 19)

    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    # Load and plot the templates and filters
    palette = itertools.cycle(sns.color_palette())
    plt.figure(figsize=(5.5, 6))
   
    for iii in range(1):
	pl = plt.scatter(x, y, s = 0.3 , color = 'k', label = str(zmin)+'<z<'+str(zmax))
        plt.ylabel(labels[iii])
	#axs[0].set_yscale('log')
        plt.xlim([14.5, 22.5])
    plt.text(16 , 0.75, r"$slope = $"+str(round(est[0],4))+"$\pm$"+str(round(est_err[0],5)))	
    plt.text(16 , 0.6 , r"$intercept = $"+str(round(est[1],4))+"$\pm$"+str(round(est_err[1],5)))	
    plt.text(16 , 0.45 , r"$scatter = $"+str(round(est[2],4))+"$\pm$"+str(round(est_err[2],5)))	

    plt.legend(loc='best')
    plt.plot(xx, pred, color="k", lw=1.5)	
    plt.ylim([-0.1, 2.1])
    plt.xlabel(r'$m_{i}$')
     
    fig_file = "/home/vakili/public_html/files/redsequence_ug/"+str(zmin)+"_z_"+str(zmax)+"fg_color_iter"+str(iteration)+".png"
    plt.savefig(fig_file)
    plt.close()

    return None


if __name__ == '__main__':

   Niter = 4  #number of iterations
   z_init = 0.06

   for i in range(11 , 12):
     
     zmin = z_init + i * 0.02
     zmax = zmin + 0.02
     x, y, yerr = test(zmin , zmax , 0)
     red_prob = mcmc(zmin , zmax , 0 , 4500 , 1000)
