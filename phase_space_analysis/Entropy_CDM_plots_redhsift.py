import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
import eagle3 as E

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

def lowess_error(x,y,frac=0.5,it=2,frac_error=0.1):
    print('Processing lowess')
    ys=lowess(y,x,frac=frac,it=2)
    sort=np.argsort(x)
    x=x[sort]
    y=y[sort]

    ys_samp=ys[0::int(1/frac_error)]

    x_samp=x[0::int(1/frac_error)]
    y_samp=y[0::int(1/frac_error)]

    error=np.empty(len(ys_samp[:,0]))
    mean_error=np.empty(len(ys_samp[:,0]))
    num=int(len(ys[:,0])*frac)
    for i in range(len(ys_samp[:,0])):
        dist=np.abs(ys_samp[i,0]-ys[:,0])
        sort_id=np.argsort(dist)
        width=dist[sort_id][num-1]
        weight=(1-(dist[sort_id][0:num-1]/width)**2)**2
        weight=np.ones(len(weight))
        val=y[sort_id][0:num-1]
        
        error[i]=(np.sum((ys_samp[i,1]-val)**2)/np.sum(weight))**0.5
        mean_error[i]=error[i]/np.sum(weight)**0.5
    return(ys,ys_samp,error,mean_error)

cross_sections=['z0','z1','z2','z3']
cross_sections2=['z0_high','z1_high','z2_high','z3_high']
num_sims=len(cross_sections)
write_loc='/hpcdata4/arisbrow/simulations/L100N256_WMAP9/Processed_data/Entropy_CDM_redshifts/'

nsims=4

colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
frac_error=np.array([1,0.01,0.01,0.01])
m=[]; params=[]; profile=[]; rad=[]; density=[]; vel_disp=[];
for i in range(nsims):
    m.append(np.load(write_loc+'Entropy_CDM_mass_'+cross_sections[i]+'.npy'))
    params.append(np.load(write_loc+'Entropy_CDM_fit_params_'+cross_sections[i]+'.npy'))
    profile.append(np.load(write_loc+'Entropy_CDM_profile_'+cross_sections[i]+'.npy'))
    rad.append(np.load(write_loc+'Rad_CDM_profile_'+cross_sections[i]+'.npy'))
    density.append(np.load(write_loc+'Entropy_CDM_density_'+cross_sections[i]+'.npy'))
    vel_disp.append(np.load(write_loc+'Entropy_CDM_vel_disp_'+cross_sections[i]+'.npy'))

m2=[]; params2=[]; profile2=[]; rad2=[]; density2=[]; vel_disp2=[];
for i in range(nsims):
    m2.append(np.load(write_loc+'Entropy_CDM_mass_'+cross_sections2[i]+'.npy'))
    params2.append(np.load(write_loc+'Entropy_CDM_fit_params_'+cross_sections2[i]+'.npy'))
    profile2.append(np.load(write_loc+'Entropy_CDM_profile_'+cross_sections2[i]+'.npy'))
    rad2.append(np.load(write_loc+'Rad_CDM_profile_'+cross_sections2[i]+'.npy'))
    density2.append(np.load(write_loc+'Entropy_CDM_density_'+cross_sections2[i]+'.npy'))
    vel_disp2.append(np.load(write_loc+'Entropy_CDM_vel_disp_'+cross_sections2[i]+'.npy'))
#identifying 'good' and 'bad' halos

good_halos=[]
bad_halos=[]
diff=[]
sigma_cut=3
for i in range(nsims):
    print(i)
    mass_cut=10**13
    diff_cut=15
    r=rad[i][m[i]>mass_cut].flatten()
    s=profile[i][m[i]>mass_cut].flatten()
    alpha,A=np.polyfit(np.log(r),np.log(s),1)

    pow_prof=np.exp(A)*rad[i]**alpha

    diff.append(np.sum(profile[i]-pow_prof,axis=1))
    mm=m[i][diff[i]<0]
    difff=diff[i][diff[i]<0]
    ms=np.empty(len(mm)*2)    
    ms[0:len(mm)]=mm
    ms[len(mm):2*len(mm)]=mm
    dif=np.empty(len(mm)*2)
    dif[0:len(mm)]=difff
    dif[len(mm):2*len(mm)]=-difff

    ys,std=lowess(dif,ms,frac=0.5,it=2,standard_deviation=True)

    std=np.interp(m[i],ms,std)
    std_diff=np.abs(diff[i]/std)

    bad_halos.append(np.where(std_diff>sigma_cut)[0])
    good_halos.append(np.where(std_diff<sigma_cut)[0]) #finding good and bad haloes

good_halos2=[]
bad_halos2=[]
diff2=[]
sigma_cut=3
for i in range(nsims):
    print(i)
    mass_cut=10**12
    diff_cut=15
    r=rad2[i][m2[i]>mass_cut].flatten()
    s=profile2[i][m2[i]>mass_cut].flatten()
    alpha,A=np.polyfit(np.log(r),np.log(s),1)

    pow_prof=np.exp(A)*rad2[i]**alpha

    diff2.append(np.sum(profile2[i]-pow_prof,axis=1))
    mm=m2[i][diff2[i]<0]
    difff=diff2[i][diff2[i]<0]
    ms=np.empty(len(mm)*2)    
    ms[0:len(mm)]=mm
    ms[len(mm):2*len(mm)]=mm
    dif=np.empty(len(mm)*2)
    dif[0:len(mm)]=difff
    dif[len(mm):2*len(mm)]=-difff

    ys,std=lowess(dif,ms,frac=0.5,it=2,standard_deviation=True)

    std=np.interp(m2[i],ms,std)
    std_diff=np.abs(diff2[i]/std)

    bad_halos2.append(np.where(std_diff>sigma_cut)[0])
    good_halos2.append(np.where(std_diff<sigma_cut)[0]) #finding good and bad haloes

#%%
plt.figure()
for i in range(nsims):
    frac=0.7
    ys=lowess(params[i][:,0][good_halos[i]],np.log(m[i][good_halos[i]]),frac=frac,it=2)
    ys2=lowess(params2[i][:,0][good_halos2[i]],np.log(m2[i][good_halos2[i]]),frac=frac,it=2)

    #plt.plot(m[i],params[i][:,0],'.',color=colors[i],alpha=0.2)
    plt.plot(np.exp(ys[:,0]),ys[:,1],'--',color=colors[i],lw=2,path_effects=[pe.Stroke(linewidth=1.2,foreground='k'),pe.Normal()],label=cross_sections[i])
    plt.plot(np.exp(ys2[:,0]),ys2[:,1],color=colors[i],lw=2,path_effects=[pe.Stroke(linewidth=1.2,foreground='k'),pe.Normal()],label=cross_sections[i])
    
    #plt.fill_between(ys_samp[:,0],ys_samp[:,1]-error,ys_samp[:,1]+error,color=colors[i],alpha=0.3)
    #plt.fill_between(ys_samp[:,0],ys_samp[:,1]-m_error,ys_samp[:,1]+m_error,color=colors[i],alpha=0.7)
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('M_200',fontsize=12)
plt.ylabel('A',fontsize=12)
plt.legend(loc='best')
plt.xlim(10**9,6*10**14)

plt.figure()
for i in range(nsims):
    frac=0.7
    ys=lowess(params[i][:,1][good_halos[i]],np.log(m[i][good_halos[i]]),frac=frac,it=2)
    ys2=lowess(params2[i][:,1][good_halos2[i]],np.log(m2[i][good_halos2[i]]),frac=frac,it=2)
    #plt.plot(m[i],params[i][:,0],'.',color=colors[i],alpha=0.2
    plt.plot(np.exp(ys[:,0]),ys[:,1],'--',color=colors[i],lw=2,path_effects=[pe.Stroke(linewidth=1.0,foreground='k'),pe.Normal()],label=cross_sections[i])
    plt.plot(np.exp(ys2[:,0]),ys2[:,1],color=colors[i],lw=2,path_effects=[pe.Stroke(linewidth=1.0,foreground='k'),pe.Normal()],label=cross_sections[i])
    #plt.fill_between(ys_samp[:,0],ys_samp[:,1]-error,ys_samp[:,1]+error,color=colors[i],alpha=0.3)
    #plt.fill_between(ys_samp[:,0],ys_samp[:,1]-m_error,ys_samp[:,1]+m_error,color=colors[i],alpha=0.7)

plt.xscale('log')
#plt.yscale('log')
plt.xlabel('M_200',fontsize=12)
plt.ylabel('$\\alpha$',fontsize=12)
plt.legend(loc='best')
plt.xlim(10**9,6*10**14)
#plt.ylim(1.5,5)

#%%
for i in range(nsims):
    fig=plt.figure()
    ax=fig.add_subplot(111)

    lc=multiline(rad[i],profile[i],np.log10(m[i]),cmap='viridis',lw=0.5,alpha=0.5)
    cb=fig.colorbar(lc)
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlim(0.03,1)
    plt.ylim(0.03,2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('r/R200',fontsize=12)
    plt.ylabel('S_calc/S_fit',fontsize=12)
    cb.set_label('log(M200)')
    plt.text(0.85,0.05,cross_sections[i],transform=ax.transAxes,fontsize=14)

plt.figure()
mean=[]
std=[]
mean2=[]
std2=[]
for i in range(nsims):
    plt.hist(params[i][:,1][good_halos[i]],histtype='step',color=colors[i],bins=50,normed=True,alpha=0.4)
    plt.hist(params2[i][:,1][good_halos2[i]],histtype='step',color=colors[i],bins=50,normed=True)
    mean.append(np.mean(params[i][:,1][good_halos[i]]))
    mean2.append(np.mean(params2[i][:,1][good_halos2[i]]))
    std.append(np.std(params[i][:,1][good_halos[i]])/len(params[i][good_halos[i]])**0.5)
    std2.append(np.std(params2[i][:,1][good_halos2[i]])/len(params2[i][good_halos2[i]])**0.5)

    plt.text(0.70,0.95-i*0.1,'%.2f $\\pm$ %.2e'%(np.mean(params[i][:,1][good_halos[i]]),np.std(params[i][:,1][good_halos[i]])/len(params[i][good_halos[i]])**0.5),color=colors[i],transform=ax.transAxes,fontsize=14)
    plt.text(0.70,0.9-i*0.1,'%.2f $\\pm$ %.2e'%(np.mean(params2[i][:,1][good_halos2[i]]),np.std(params2[i][:,1][good_halos2[i]])/len(params2[i][good_halos2[i]])**0.5),color=colors[i],transform=ax.transAxes,fontsize=14)

plt.ylabel('Prob Density')
plt.xlabel('Exponent of power law')


print('z1: %.2f'%(np.abs(mean[0]-mean[1])/std[1]))
print('z1: %.2f'%(np.abs(mean2[0]-mean2[1])/std2[1]))
print('z2: %.2f'%(np.abs(mean[0]-mean[2])/std[2]))
print('z2: %.2f'%(np.abs(mean2[0]-mean2[2])/std2[2]))
print('z3: %.2f'%(np.abs(mean[0]-mean[3])/std[3]))
print('z3: %.2f'%(np.abs(mean2[0]-mean2[3])/std2[3]))
plt.show()