import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.collections import LineCollection

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

m=np.load('Entropy_mass_0.0.npy')
m_0=np.load('Entropy_mass_0.1.npy')
m_1=np.load('Entropy_mass_1.0.npy')
m_5=np.load('Entropy_mass_5.0.npy')
m_10=np.load('Entropy_mass_10.0.npy')

params=np.load('Entropy_fit_params_0.0.npy')
params_0=np.load('Entropy_fit_params_0.1.npy')
params_1=np.load('Entropy_fit_params_1.0.npy')
params_5=np.load('Entropy_fit_params_5.0.npy')
params_10=np.load('Entropy_fit_params_10.0.npy')

profile=np.load('Entropy_profile_0.0.npy')
profile_0=np.load('Entropy_profile_0.1.npy')
profile_1=np.load('Entropy_profile_1.0.npy')
profile_5=np.load('Entropy_profile_5.0.npy')
profile_10=np.load('Entropy_profile_10.0.npy')

rad=np.load('Rad_profile_0.0.npy')
rad_0=np.load('Rad_profile_0.1.npy')
rad_1=np.load('Rad_profile_1.0.npy')
rad_5=np.load('Rad_profile_5.0.npy')
rad_10=np.load('Rad_profile_10.0.npy')

#%%
frac=0.6
ys=lowess(params[:,0],m,frac=frac,it=3)
ys_0=lowess(params_0[:,0],m_0,frac=frac,it=3)
ys_1=lowess(params_1[:,0],m_1,frac=frac,it=3)
ys_2=lowess(params_5[:,0],m_5,frac=frac,it=3)
ys_3=lowess(params_10[:,0],m_10,frac=frac,it=3)
plt.figure()
plt.plot(m,params[:,0],'.',color='#1f77b4',alpha=0.2)
plt.plot(m_0,params_0[:,0],'.',color='#ff7f0e',alpha=0.2)
plt.plot(m_1,params_1[:,0],'.',color='#2ca02c',alpha=0.2)
plt.plot(m_5,params_5[:,0],'.',color='#d62728',alpha=0.2)
plt.plot(m_10,params_10[:,0],'.',color='#9467bd',alpha=0.2)
plt.plot(ys[:,0],ys[:,1],color='#1f77b4',lw=2,label='$\\sigma=0$')
plt.plot(ys_0[:,0],ys_0[:,1],color='#ff7f0e',lw=2,label='$\\sigma=0.1$')
plt.plot(ys_1[:,0],ys_1[:,1],color='#2ca02c',lw=2,label='$\\sigma=1$')
plt.plot(ys_2[:,0],ys_2[:,1],color='#d62728',lw=2,label='$\\sigma=5$')
plt.plot(ys_3[:,0],ys_3[:,1],color='#9467bd',lw=2,label='$\\sigma=10$')
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('r/R200',fontsize=12)
plt.ylabel('A',fontsize=12)
plt.legend(loc='best')
plt.xlim(10**12,6*10**14)
plt.ylim(1,5)


#%%
ys=lowess(params[:,1],m,frac=frac,it=3)
ys_0=lowess(params_0[:,1],m_0,frac=frac,it=3)
ys_1=lowess(params_1[:,1],m_1,frac=frac,it=3)
ys_2=lowess(params_5[:,1],m_5,frac=frac,it=3)
ys_3=lowess(params_10[:,1],m_10,frac=frac,it=3)
plt.figure()
plt.plot(m,params[:,1],'.',color='#1f77b4',alpha=0.2)
plt.plot(m_0,params_0[:,1],'.',color='#ff7f0e',alpha=0.2)
plt.plot(m_1,params_1[:,1],'.',color='#2ca02c',alpha=0.2)
plt.plot(m_5,params_5[:,1],'.',color='#d62728',alpha=0.2)
plt.plot(m_10,params_10[:,1],'.',color='#9467bd',alpha=0.2)
plt.plot(ys[:,0],ys[:,1],color='#1f77b4',lw=2,label='$\\sigma=0$')
plt.plot(ys_0[:,0],ys_0[:,1],color='#ff7f0e',lw=2,label='$\\sigma=0.1$')
plt.plot(ys_1[:,0],ys_1[:,1],color='#2ca02c',lw=2,label='$\\sigma=1$')
plt.plot(ys_2[:,0],ys_2[:,1],color='#d62728',lw=2,label='$\\sigma=5$')
plt.plot(ys_3[:,0],ys_3[:,1],color='#9467bd',lw=2,label='$\\sigma=10$')
plt.plot(ys[:,0],np.ones(len(ys[:,0]))*(1.875*2/3),'k--')
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('r/R200',fontsize=12)
plt.ylabel('$\\alpha$',fontsize=12)
plt.legend(loc='best')
plt.xlim(10**12,6*10**14)
plt.ylim(0,2.5)

#%%
ys=lowess(params[:,2],m,frac=frac,it=3)
ys_0=lowess(params_0[:,2],m_0,frac=frac,it=3)
ys_1=lowess(params_1[:,2],m_1,frac=frac,it=3)
ys_2=lowess(params_5[:,2],m_5,frac=frac,it=3)
ys_3=lowess(params_10[:,2],m_10,frac=frac,it=3)
plt.figure()
plt.plot(m,params[:,2],'.',color='#1f77b4',alpha=0.2)
plt.plot(m_0,params_0[:,2],'.',color='#ff7f0e',alpha=0.2)
plt.plot(m_1,params_1[:,2],'.',color='#2ca02c',alpha=0.2)
plt.plot(m_5,params_5[:,2],'.',color='#d62728',alpha=0.2)
plt.plot(m_10,params_10[:,2],'.',color='#9467bd',alpha=0.2)
plt.plot(ys[:,0],ys[:,1],color='#1f77b4',lw=2,label='$\\sigma=0$')
plt.plot(ys_0[:,0],ys_0[:,1],color='#ff7f0e',lw=2,label='$\\sigma=0.1$')
plt.plot(ys_1[:,0],ys_1[:,1],color='#2ca02c',lw=2,label='$\\sigma=1$')
plt.plot(ys_2[:,0],ys_2[:,1],color='#d62728',lw=2,label='$\\sigma=5$')
plt.plot(ys_3[:,0],ys_3[:,1],color='#9467bd',lw=2,label='$\\sigma=10$')
# norm=ys_1[-1,1]/(ys_1[-1,0]**(3/4*2/3))
# plt.plot(ys_1[:,0],norm*ys_1[:,0]**(3/4*2/3),'k--',alpha=0.8)
# norm=ys_2[-1,1]/(ys_2[-1,0]**(3/4*2/3))
# plt.plot(ys_2[:,0],norm*ys_2[:,0]**(3/4*2/3),'k--',alpha=0.8)
# norm=ys_3[-1,1]/(ys_3[-1,0]**(3/4*2/3))
# plt.plot(ys_3[:,0],norm*ys_3[:,0]**(3/4*2/3),'k--',alpha=0.8)
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('r/R200',fontsize=12)
plt.ylabel('S$_{\\rm{core}}$',fontsize=12)
plt.legend(loc='best')
plt.xlim(10**12,6*10**14)
plt.ylim(-1,0.5)


#%%
params_adj=np.interp(m,ys[:,0],ys[:,1])
params_adj_0=np.interp(m_0,ys[:,0],ys[:,1])
params_adj_1=np.interp(m_1,ys[:,0],ys[:,1])
params_adj_5=np.interp(m_5,ys[:,0],ys[:,1])
params_adj_10=np.interp(m_10,ys[:,0],ys[:,1])

ys=lowess(params[:,2]-params_adj,m,frac=frac,it=1)
ys_0=lowess(params_0[:,2]-params_adj_0,m_0,frac=frac,it=3)
ys_1=lowess(params_1[:,2]-params_adj_1,m_1,frac=frac,it=3)
ys_2=lowess(params_5[:,2]-params_adj_5,m_5,frac=frac,it=3)
ys_3=lowess(params_10[:,2]-params_adj_10,m_10,frac=frac,it=3)
plt.figure()
plt.plot(m,params[:,2]-params_adj,'.',color='#1f77b4',alpha=0.2)
plt.plot(m_0,params_0[:,2]-params_adj_0,'.',color='#ff7f0e',alpha=0.2)
plt.plot(m_1,params_1[:,2]-params_adj_1,'.',color='#2ca02c',alpha=0.2)
plt.plot(m_5,params_5[:,2]-params_adj_5,'.',color='#d62728',alpha=0.2)
plt.plot(m_10,params_10[:,2]-params_adj_10,'.',color='#9467bd',alpha=0.2)
plt.plot(ys[:,0],ys[:,1],color='#1f77b4',lw=2,label='$\\sigma=0$')
plt.plot(ys_0[:,0],ys_0[:,1],color='#ff7f0e',lw=2,label='$\\sigma=0.1$')
plt.plot(ys_1[:,0],ys_1[:,1],color='#2ca02c',lw=2,label='$\\sigma=1$')
plt.plot(ys_2[:,0],ys_2[:,1],color='#d62728',lw=2,label='$\\sigma=5$')
plt.plot(ys_3[:,0],ys_3[:,1],color='#9467bd',lw=2,label='$\\sigma=10$')
# norm=ys_1[-1,1]/(ys_1[-1,0]**(3/4*2/3))
# plt.plot(ys_1[:,0],norm*ys_1[:,0]**(3/4*2/3),'k--',alpha=0.8)
# norm=ys_2[-1,1]/(ys_2[-1,0]**(3/4*2/3))
# plt.plot(ys_2[:,0],norm*ys_2[:,0]**(3/4*2/3),'k--',alpha=0.8)
# norm=ys_3[-1,1]/(ys_3[-1,0]**(3/4*2/3))
# plt.plot(ys_3[:,0],norm*ys_3[:,0]**(3/4*2/3),'k--',alpha=0.8)
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('M200',fontsize=12)
plt.ylabel('S$_{\\rm{core}}$',fontsize=12)
plt.legend(loc='best')
plt.xlim(10**12,6*10**14)
plt.ylim(-1,0.5)


mass_cut=10**16
#%%
fig=plt.figure()
ax=fig.add_subplot(111)

lc=multiline(rad[m<mass_cut],profile[m<mass_cut],np.log10(m),cmap='viridis',lw=0.5,alpha=0.5)
cb=fig.colorbar(lc)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.03,1)
plt.ylim(0.02,10)
plt.xlabel('r/R200',fontsize=12)
plt.ylabel('S',fontsize=12)
cb.set_label('log(M200)')
plt.text(0.85,0.05,'$\\sigma=0$',transform=ax.transAxes,fontsize=14)

fig=plt.figure()
ax=fig.add_subplot(111)

lc=multiline(rad_0[m_0<mass_cut],profile_0[m_0<mass_cut],np.log10(m_0),cmap='viridis',lw=0.5,alpha=0.5)
cb=fig.colorbar(lc)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.03,1)
plt.ylim(0.02,10)
plt.xlabel('r/R200',fontsize=12)
plt.ylabel('S',fontsize=12)
cb.set_label('log(M200)')
plt.text(0.85,0.05,'$\\sigma=0.1$',transform=ax.transAxes,fontsize=14)

fig=plt.figure()
ax=fig.add_subplot(111)

lc=multiline(rad_1[m_1<mass_cut],profile_1[m_1<mass_cut],np.log10(m_1),cmap='viridis',lw=0.5,alpha=0.5)
cb=fig.colorbar(lc)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.03,1)
plt.ylim(0.02,10)
plt.xlabel('r/R200',fontsize=12)
plt.ylabel('S',fontsize=12)
cb.set_label('log(M200)')
plt.text(0.85,0.05,'$\\sigma=1$',transform=ax.transAxes,fontsize=14)

fig=plt.figure()
ax=fig.add_subplot(111)

lc=multiline(rad_5[m_5<mass_cut],profile_5[m_5<mass_cut],np.log10(m_5),cmap='viridis',lw=0.5,alpha=0.5)
cb=fig.colorbar(lc)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.03,1)
plt.ylim(0.02,10)
plt.xlabel('r/R200',fontsize=12)
plt.ylabel('S',fontsize=12)
cb.set_label('log(M200)')
plt.text(0.85,0.05,'$\\sigma=5$',transform=ax.transAxes,fontsize=14)

fig=plt.figure()
ax=fig.add_subplot(111)

lc=multiline(rad_10[m_10<mass_cut],profile_10[m_10<mass_cut],np.log10(m_10),cmap='viridis',lw=0.5,alpha=0.5)
cb=fig.colorbar(lc)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.03,1)
plt.ylim(0.02,10)
plt.xlabel('r/R200',fontsize=12)
plt.ylabel('S',fontsize=12)
cb.set_label('log(M200)')
plt.text(0.85,0.05,'$\\sigma=10$',transform=ax.transAxes,fontsize=14)
plt.show()