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

nsims=2
cross_sections=['CDM_low_norm_bins','CDM_low_high_bins']
cross_sections2=['z0','z0_high']
colors=['#1f77b4','#ff7f0e']#,'#2ca02c','#d62728']

write_loc='/hpcdata4/arisbrow/simulations/L100N256_WMAP9/Processed_data/Entropy_CDM_bin_test/'
write_loc2='/hpcdata4/arisbrow/simulations/L100N256_WMAP9/Processed_data/Entropy_CDM_redshifts/'
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
    m2.append(np.load(write_loc2+'Entropy_CDM_mass_'+cross_sections2[i]+'.npy'))
    params2.append(np.load(write_loc2+'Entropy_CDM_fit_params_'+cross_sections2[i]+'.npy'))
    profile2.append(np.load(write_loc2+'Entropy_CDM_profile_'+cross_sections2[i]+'.npy'))
    rad2.append(np.load(write_loc2+'Rad_CDM_profile_'+cross_sections2[i]+'.npy'))
    density2.append(np.load(write_loc2+'Entropy_CDM_density_'+cross_sections2[i]+'.npy'))
    vel_disp2.append(np.load(write_loc2+'Entropy_CDM_vel_disp_'+cross_sections2[i]+'.npy'))

#identifying 'good' and 'bad' halos
diff=[]

for i in range(nsims):
    mass_cut=10**13
    diff_cut=15
    r=rad[i][m[i]>mass_cut].flatten()
    s=profile[i][m[i]>mass_cut].flatten()
    alpha,A=np.polyfit(np.log(r),np.log(s),1)

    pow_prof=np.exp(A)*rad[i]**alpha

    diff.append(np.sum(profile[i]-pow_prof,axis=1))

diff2=[]
for i in range(nsims):
    mass_cut=10**13
    diff_cut=15
    r=rad2[i][m2[i]>mass_cut].flatten()
    s=profile[i][m[i]>mass_cut].flatten()
    alpha,A=np.polyfit(np.log(r),np.log(s),1)

    pow_prof=np.exp(A)*rad2[i]**alpha

    diff2.append(np.sum(profile2[i]-pow_prof,axis=1))

plt.figure()
ax=plt.subplot(111)
plt.scatter(m[1],diff[1],s=2)#,c=np.log(subhalo_mass_ratio),alpha=0.7)
plt.text(0.65,0.05,'Normal bins',transform=ax.transAxes,fontsize=14)
print(len(diff[1][diff[1]>500])/len(diff[1]))
#plt.colorbar()
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('M200',fontsize=14)
plt.ylabel('S-S_fit',fontsize=14)
plt.xlim(10**9.5,10**14)
plt.ylim(-50,2000)

plt.figure()
ax=plt.subplot(111)
plt.scatter(m2[1],diff2[1],s=2)#,c=np.log(subhalo_mass_ratio),alpha=0.7)
plt.text(0.65,0.05,'Gaussian kernel',transform=ax.transAxes,fontsize=14)
print(len(diff[1][diff[1]>500])/len(diff[1]))
#plt.colorbar()
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('M200',fontsize=14)
plt.ylabel('S-S_fit',fontsize=14)
plt.xlim(10**9.5,10**14)
plt.ylim(-50,2000)



fig=plt.figure()
ax=fig.add_subplot(111)

lc=multiline(rad[1],profile[1],np.log10(m[1]),cmap='viridis',lw=0.5,alpha=0.5)
cb=fig.colorbar(lc)
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-2.5,10**0)
plt.ylim(10**-3,10**3)
plt.xlabel('r/R200',fontsize=12)
plt.ylabel('$\\rho$',fontsize=12)
cb.set_label('log(M200)')
plt.text(0.65,0.05,'Normal bins',transform=ax.transAxes,fontsize=14)


fig=plt.figure()
ax=fig.add_subplot(111)

lc=multiline(rad2[1],profile2[1],np.log10(m2[1]),cmap='viridis',lw=0.5,alpha=0.5)
cb=fig.colorbar(lc)
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-2.5,10**0)
plt.ylim(10**-3,10**3)
plt.xlabel('r/R200',fontsize=12)
plt.ylabel('$\\sigma$',fontsize=12)
cb.set_label('log(M200)')
plt.text(0.65,0.05,'Gaussian kernel',transform=ax.transAxes,fontsize=14)

plt.show()
exit()
#calculating the tidal forces on a halo
# sim='/hpcdata4/sam/PhD/Investigating_Running/RUNS/DMONLY/L025N1024/run_0/data'
# tag='033'
# M200 = E.readArray("SUBFIND_GROUP", sim, tag, "FOF/Group_M_Crit200",noH=False); M200*=10**10
# R200 = E.readArray("SUBFIND_GROUP", sim, tag, "FOF/Group_R_Crit200",noH=False)
# loc=E.readArray("SUBFIND_GROUP", sim, tag, "FOF/GroupCentreOfPotential",noH=False)

# num_sub = E.readArray("SUBFIND_GROUP", sim, tag, "FOF/NumOfSubhalos",noH=False);
# sub_id = E.readArray("SUBFIND_GROUP", sim, tag, "FOF/FirstSubhaloID",noH=False);

# sub_mass = E.readArray("SUBFIND",sim,tag,"Subhalo/Mass",noH=False); 
# sub_loc = E.readArray("SUBFIND",sim,tag,"/Subhalo/CentreOfPotential"); 

# sub_rad = E.readArray("SUBFIND",sim,tag,"/Subhalo/HalfMassRad",noH=False)[:,1]; 

# sort=np.argsort(M200)
# M200=M200[sort]
# R200=R200[sort]
# loc=loc[sort]
# num_sub=num_sub[sort]
# sub_id=sub_id[sort]

# mas_cut=10**9
# cut=np.where(M200>mas_cut)[0]
# M200=M200[cut]
# R200=R200[cut]
# loc=loc[cut]
# num_sub=num_sub[cut]
# sub_id=sub_id[cut]


# cut=np.where(m[1]>mas_cut)[0]
# m[1]=m[1][cut]
# params[1]=params[1][cut]
# profile[1]=profile[1][cut]
# rad[1]=rad[1][cut]


# summed_mass=np.empty(len(num_sub))

# for i in range(len(num_sub)):
#     if num_sub[i]==1:
#         summed_mass[i]=-1
#         continue
#     mass=sub_mass[sub_id[i]:sub_id[i]+num_sub[i]]

#     rat=mass[1:]/mass[0]
#     summed_mass[i]=np.sum(mass[1:])/mass[0]


# tid_force=np.empty(len(M200))
# for i in range(len(M200)):
#     dist=((loc[i,0]-loc[:,0])**2+(loc[i,1]-loc[:,1])**2+(loc[i,2]-loc[:,2])**2)**0.5
#     dist[dist==0]=100
#     rat=M200/M200[i]*R200[i]**2/dist**2
#     tid_force[i]=np.max(rat)

# subhalo_mass_ratio=np.empty(len(M200))
# print(np.sum(num_sub==0))

# for i in range(len(M200)):
#     if num_sub[i]==0:
#         subhalo_mass_ratio[i]=-1.0
#         continue
#     elif num_sub[i]==1:
#         subhalo_mass_ratio[i]=0.0
#         continue
#     rad_host =  sub_rad[sub_id[i]]
#     mass=sub_mass[sub_id[i]:sub_id[i]+num_sub[i]]
#     loc=sub_loc[sub_id[i]:sub_id[i]+num_sub[i]]

#     mass_host=mass[0]
#     mass_sub=mass[1:]
    
#     loc_host=loc[0]
#     loc_sub=loc[1:]
#     dist=((loc_sub[:,0]-loc_host[0])**2+(loc_sub[:,1]-loc_host[1])**2+(loc_sub[:,2]-loc_host[2])**2)**0.5

#     tid_forc_rat=4*mass_sub/mass_host*dist*rad_host**3/((dist+rad_host)**2*(dist-rad_host)**2)
    
#     subhalo_mass_ratio[i]=np.max(tid_forc_rat)
# #calculating residuals from a power law fit of the most massive haloes
# mass_cut=10**13
# r=rad[1][m[1]>mass_cut].flatten()
# s=profile[1][m[1]>mass_cut].flatten()
# alpha,A=np.polyfit(np.log(r),np.log(s),1)

# pow_prof=np.exp(A)*rad[1]**alpha

# diff=np.sum(profile[1]-pow_prof,axis=1)

# plt.figure()
# plt.scatter(M200,diff,s=2)#,c=np.log(subhalo_mass_ratio),alpha=0.7)

# print(len(diff[(M200<2.3*10**10) & (diff>500)])/len(diff[M200<2.3*10**10]))
# #plt.colorbar()
# plt.xscale('log')
# #plt.yscale('log')
# plt.xlabel('M200',fontsize=14)
# plt.ylabel('S-S_fit',fontsize=14)

# bad_halos=np.where(diff>1000)
# good_halos=np.where(diff<1000)

# print(np.max(summed_mass))
# plt.figure()
# plt.plot(M200[good_halos],R200[good_halos],'k.')
# plt.plot(M200[bad_halos],R200[bad_halos],'b.')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()