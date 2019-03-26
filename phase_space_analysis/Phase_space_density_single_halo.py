import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

def var_func(vel):
	var=np.var(vel)
	return(var)
def loader(sidm):

	#loading in particle data for each group
	if sidm==0:
		particle_sub=h5.File('/local/arisbrow/SIDM_TESTS/L100N256_WMAP9/DMONLY/data/particledata_034/eagle_subfind_particles_034.0.hdf5')
	elif sidm==0.1:
		particle_sub=h5.File('/local/arisbrow/SIDM_TESTS/L100N256_WMAP9/DMONLY_SIDM0.1/data/particledata_034/eagle_subfind_particles_034.0.hdf5')
	elif sidm==1:
		particle_sub=h5.File('/local/arisbrow/SIDM_TESTS/L100N256_WMAP9/DMONLY_SIDM1/data/particledata_034/eagle_subfind_particles_034.0.hdf5')
	
	part_mass=particle_sub['Header'].attrs['MassTable'][1]
	group_id=particle_sub['PartType1']['GroupNumber'][()]
	group_id=np.abs(group_id)-1
	part_pos=particle_sub['PartType1']['Coordinates'][()]
	part_vel=particle_sub['PartType1']['Velocity'][()]
	sort=np.argsort(group_id)
	group_id=group_id[sort]
	part_pos=part_pos[sort]
	part_vel=part_vel[sort]
	sub,ind,count=np.unique(group_id,return_index=True,return_counts=True)

	#loading in group properties
	if sidm==0:
		h=h5.File('/local/arisbrow/SIDM_TESTS/L100N256_WMAP9/DMONLY/data/groups_034/eagle_subfind_tab_034.0.hdf5')
	elif sidm==0.1:
		h=h5.File('/local/arisbrow/SIDM_TESTS/L100N256_WMAP9/DMONLY_SIDM0.1/data/groups_034/eagle_subfind_tab_034.0.hdf5')
	elif sidm==1:
		h=h5.File('/local/arisbrow/SIDM_TESTS/L100N256_WMAP9/DMONLY_SIDM1/data/groups_034/eagle_subfind_tab_034.0.hdf5')
	M_200=h['FOF']['Group_M_Crit200'][()]
	R_200=h['FOF']['Group_R_Crit200'][()]
	group_pos=h['FOF']['GroupCentreOfPotential'][()]
	for i in range(31):
		if sidm==0:
			h=h5.File('/local/arisbrow/SIDM_TESTS/L100N256_WMAP9/DMONLY/data/groups_034/eagle_subfind_tab_034.'+str(i+1)+'.hdf5')
		elif sidm==0.1:
			h=h5.File('/local/arisbrow/SIDM_TESTS/L100N256_WMAP9/DMONLY_SIDM0.1/data/groups_034/eagle_subfind_tab_034.'+str(i+1)+'.hdf5')
		elif sidm==1:
			h=h5.File('/local/arisbrow/SIDM_TESTS/L100N256_WMAP9/DMONLY_SIDM1/data/groups_034/eagle_subfind_tab_034.'+str(i+1)+'.hdf5')
		M_200=np.append(M_200,h['FOF']['Group_M_Crit200'][()])
		R_200=np.append(R_200,h['FOF']['Group_R_Crit200'][()])
		group_pos=np.append(group_pos,h['FOF']['GroupCentreOfPotential'][()],axis=0)

	return(ind,count,M_200,R_200,group_pos,part_pos,part_vel,part_mass)

def radial_profile(pos,group_pos,R_2,h,part_mass,nbin=50,outer_rad=1):
	r=((pos[:,0]-group_pos[0])**2+(pos[:,1]-group_pos[1])**2+(pos[:,2]-group_pos[2])**2)**0.5

	r=np.sort(r)

	nbins=50
	r_sample=np.logspace(np.log10(s_length),np.log10(outer_rad*R_2-s_length),nbins)
	dmdr=np.empty(len(r_sample))
	
	norm=h*np.pi**0.5
	for i in range(nbins):
		weight=np.exp(-((r_sample[i]-r)/h)**2)
		norm=np.trapz(weight,r)
		dmdr[i]=np.sum(weight)*part_mass/norm

	rho=dmdr/(4*np.pi*r_sample**2)
	return(r_sample,rho)

def dens_calc(pos,vel,loc,num_p,part_mass):
	dist=((pos[:,0]-loc[0])**2+(pos[:,1]-loc[1])**2+(pos[:,2]-loc[2])**2)**0.5
	h=np.sort(dist)[num_p]
	dist_half=dist[dist/h<0.5]/h
	vel_half=vel[dist/h<0.5]
	dist_one=dist[(dist/h>0.5) & (dist/h<1)]/h
	vel_one=vel[(dist/h>0.5) & (dist/h<1)]
	weight1=1-6*dist_half**2+6*dist_half**3
	weight2=2*(1-dist_one)**3
	weight_sum=np.sum(weight1)+np.sum(weight2)
	norm=8/(np.pi*h**3)
	rho=(np.sum(weight1)+np.sum(weight2))*part_mass*norm

	vel_mean1=np.sum((vel_half.transpose()*weight1).transpose(),axis=0)
	vel_mean2=np.sum((vel_one.transpose()*weight2).transpose(),axis=0)
	vel_mean=(vel_mean1+vel_mean2)/(weight_sum)

	vel_disp1=vel_half-vel_mean
	vel_disp2=vel_one-vel_mean

	vel_disp1=((vel_disp1.transpose()*weight1).transpose())**2
	vel_disp2=((vel_disp2.transpose()*weight2).transpose())**2
	
	vel_dispx=(np.sum(vel_disp1[:,0])+np.sum(vel_disp2[:,0]))
	vel_dispy=(np.sum(vel_disp1[:,1])+np.sum(vel_disp2[:,1]))
	vel_dispz=(np.sum(vel_disp1[:,2])+np.sum(vel_disp2[:,2]))

	
	vel_disp=((vel_dispx+vel_dispy+vel_dispz)/weight_sum)**0.5
	return(rho,vel_mean,vel_disp)

def NFW(r,R_s,rho_0):
	return(np.log(rho_0/(r/R_s*(1+r/R_s)**2)))

def phase_bin_constn(r,vel_s,R200,m,var_mass=False):

	nparts=int(0.339*len(r)**0.79)
	
	vel_s=vel_s[np.where(r<R200)]
	if var_mass==True:
		m=m[np.where(r<R200)]
	r=r[np.where(r<R200)]
	r=np.sort(r)
    
	n_bin_edges=int(len(r)/nparts)
    
	bin_edges=np.empty(n_bin_edges)
	for i in range(n_bin_edges):
		bin_edges[i]=r[i*nparts]
        
	bin_edges[n_bin_edges-1]=r[len(r)-1]
	nbins=n_bin_edges-1
	bin_centre=np.empty(nbins)
	vol=np.empty(nbins)
	for i in range(nbins):
		bin_centre[i]=((bin_edges[i+1]**2+bin_edges[i]**2)/2)**0.5
		vol[i]=4/3*np.pi*(bin_edges[i+1]**3-bin_edges[i]**3)

	if var_mass==False:
		dmmdr=np.histogram(r,bin_edges)[0]
		rho_bin=dmmdr*m/vol
	elif var_mass==True:
		dmmdr=stats.binned_statistic(r,m,'sum',bin_edges)[0]
		rho_bin=dmmdr/vol

	var_r=stats.binned_statistic(r,vel_s[:,0],var_func,bin_edges)[0]
	var_th=stats.binned_statistic(r,vel_s[:,1],var_func,bin_edges)[0]
	var_ph=stats.binned_statistic(r,vel_s[:,2],var_func,bin_edges)[0]
	var=np.empty(len(var_r),3)
	var[:,0]=var_r**0.5
	var[:,1]=var_th**0.5
	var[:,2]=var_ph**0.5
	return(rho_bin,var,bin_centre)

def phase_bin_log(r,vel_s,R200,nbin,m,var_mass=False):

	nparts=int(0.339*np.sum(r<R200)**0.79)
	
	vel_s=vel_s[np.where(r<R200)]
	if var_mass==True:
		m=m[np.where(r<R200)]
	r=r[np.where(r<R200)]
	
	bin_edges=np.logspace(np.log10(0.004),np.log10(R200),nbin)
        
	
	nbins=nbin-1
	bin_centre=np.empty(nbins)
	vol=np.empty(nbins)
	for i in range(nbins):
		bin_centre[i]=(bin_edges[i+1]+bin_edges[i])/2
		vol[i]=4/3*np.pi*(bin_edges[i+1]**3-bin_edges[i]**3)

	if var_mass==False:
		dmmdr=np.histogram(r,bin_edges)[0]
		rho_bin=dmmdr*m/vol
	elif var_mass==True:
		dmmdr=stats.binned_statistic(r,m,'sum',bin_edges)[0]
		rho_bin=dmmdr/vol

	var_r=stats.binned_statistic(r,vel_s[:,0],var_func,bin_edges)[0]
	var_th=stats.binned_statistic(r,vel_s[:,1],var_func,bin_edges)[0]
	var_ph=stats.binned_statistic(r,vel_s[:,2],var_func,bin_edges)[0]
	var=np.empty((len(var_r),3))
	var[:,0]=var_r**0.5
	var[:,1]=var_th**0.5
	var[:,2]=var_ph**0.5
	return(rho_bin,var,bin_centre)

def phase_gaussian_kernel(r,vel,R200,bin_centre,m,var_mass=False):

	vel=vel[np.where(r<1.5*R200)]
	if var_mass==True:
		m=m[np.where(r<1.5*R200)]
	r=r[np.where(r<1.5*R200)]

	num_p=int(3.146*np.sum(r<R200)**(0.79))*2
	rho=np.empty(len(bin_centre))
	vel_dd=np.empty((len(bin_centre),3))
	nnparts=np.ones(len(bin_centre))*num_p

	for i in range(len(bin_centre)):
		dist=np.abs(r-bin_centre[i])
		h=np.sort(dist)[num_p-1]
		if h>bin_centre[i]:
			h=bin_centre[i]
			nnparts[i]=np.sum(np.abs(r-bin_centre[i])<dist)

		
		dist_half=dist[dist/h<0.5]/h
		dist_one=dist[(dist/h>0.5) & (dist/h<1)]/h
		vel_half=vel[dist/h<0.5]
		vel_one=vel[(dist/h>0.5) & (dist/h<1)]
		if var_mass==True:
			m_half=m[dist/h<0.5]
			m_one=m[(dist/h>0.5) & (dist/h<1)]

		norm=(8/np.pi)**(1/3)/h
		vol_norm=(norm)*4*np.pi*h*(1/16*h**2+3/4*bin_centre[i]**2)
		weight1=(1-6*dist_half**2+6*dist_half**3)*norm
		weight2=2*(1-dist_one)**3*norm
		weight_sum=np.sum(weight1)+np.sum(weight2)
		
		if var_mass==False:
			dmdr=(np.sum(weight1*m)+np.sum(weight2*m))
		elif var_mass==True:
			dmdr=(np.sum(weight1*m_half)+np.sum(weight2*m_one))
		rho[i]=dmdr/(vol_norm)

		vel_meanx=(np.sum(weight1*vel_half[:,0])+np.sum(weight2*vel_one[:,0]))/weight_sum
		vel_meany=(np.sum(weight1*vel_half[:,1])+np.sum(weight2*vel_one[:,1]))/weight_sum
		vel_meanz=(np.sum(weight1*vel_half[:,2])+np.sum(weight2*vel_one[:,2]))/weight_sum

		vel_dispx=(np.sum(weight1*(vel_half[:,0]-vel_meanx)**2)+np.sum(weight2*(vel_one[:,0]-vel_meanx)**2))/weight_sum
		vel_dispy=(np.sum(weight1*(vel_half[:,1]-vel_meany)**2)+np.sum(weight2*(vel_one[:,1]-vel_meany)**2))/weight_sum
		vel_dispz=(np.sum(weight1*(vel_half[:,2]-vel_meanz)**2)+np.sum(weight2*(vel_one[:,2]-vel_meanz)**2))/weight_sum

		vel_dd[i,0]=vel_dispx**0.5
		vel_dd[i,1]=vel_dispy**0.5
		vel_dd[i,2]=vel_dispz**0.5
	return(rho,vel_dd,bin_centre)

def cartesian_to_spherical(pos,vel):
	pos[np.where(pos==0.0)]=0.0001 #
	pos_spherical=np.empty(pos.shape)
	pos_spherical[:,0]=np.sqrt(pos[:,0]**2+pos[:,1]**2+pos[:,2]**2)
	pos_spherical[:,1]=np.arccos(pos[:,2]/pos_spherical[:,0])
	pos_spherical[:,2]=np.arctan(pos[:,1]/pos[:,0])

	vel_spherical=np.empty(pos.shape)
	vel_spherical[:,0]=vel[:,0]*np.sin(pos_spherical[:,1])*np.cos(pos_spherical[:,2])+vel[:,1]*np.sin(pos_spherical[:,1])*np.sin(pos_spherical[:,2])+vel[:,2]*np.cos(pos_spherical[:,1])
	vel_spherical[:,1]=vel[:,0]*np.cos(pos_spherical[:,1])*np.cos(pos_spherical[:,2])+vel[:,1]*np.cos(pos_spherical[:,1])*np.sin(pos_spherical[:,2])-vel[:,2]*np.sin(pos_spherical[:,1])
	vel_spherical[:,2]=-vel[:,0]*np.sin(pos_spherical[:,2])+vel[:,1]*np.cos(pos_spherical[:,2])
	return(pos_spherical,vel_spherical)

def stacker(stack_id,part_pos,part_vel,ind,M200,R200,group_pos,count,part_mass):
	total_num=np.sum(count[stack_id])
	pos=np.empty((total_num,3))
	vel=np.empty((total_num,3))
	mass=np.empty(total_num)
	counter=0
	for i in range(len(stack_id)):
		print(i)
		group_num=stack_id[i]
		poss=part_pos[ind[group_num]:ind[group_num]+count[group_num]]
		vels=part_vel[ind[group_num]:ind[group_num]+count[group_num]]
		m2=M200[group_num]
		r2=R200[group_num]
		poss[:,0]-=group_pos[group_num,0]; poss[:,1]-=group_pos[group_num,1]; poss[:,2]-=group_pos[group_num,2]
		r=np.sqrt(poss[:,0]**2+poss[:,1]**2+poss[:,2]**2)
		r_less=np.where(r<r2)
		
		vel_x=np.mean(vels[:,0][r_less]); vel_y=np.mean(vels[:,1][r_less]); vel_z=np.mean(vels[:,2][r_less])
		vels[:,0]-=vel_x; vels[:,1]-=vel_y; vels[:,2]-=vel_z

		v_circ2=6.557*10**(-5)*np.sqrt(m2/r2)
		pos[counter:counter+count[group_num],:]=poss/r2
		vel[counter:counter+count[group_num],:]=vels/v_circ2
		mass[counter:counter+count[group_num]]=np.ones(count[group_num])*part_mass/m2
		counter+=count[group_num]
	
	return(pos,vel,mass)

def mass_prof(r,m,bin_centre):
	mas_r=np.empty(len(bin_centre))
	for i in range(len(bin_centre)):
		mas_r[i]=np.sum(m[r<bin_centre[i]])
	return(mas_r)

def derivative(x,y):
    dydx=np.empty(len(x))

    for i in range(len(x)-2):
        h2=x[i+1]-x[i]
        h1=x[i+2]-x[i+1]
        dydx[i+1]=-h1/(h2*(h1+h2))*y[i]+(h1-h2)/(h1*h2)*y[i+1]+h2/(h1*(h1+h2))*y[i+2]
        
    dydx[0]=(y[1]-y[0])/(x[1]-x[0])
    dydx[len(x)-1]=(y[len(x)-1]-y[len(x)-2])/(x[len(x)-1]-x[len(x)-2])
    return(dydx)
s_length=0.004 #gravitational softening length of the sim




nbins=50
ind,count,M200,R200,group_pos,part_pos,part_vel,part_mass=loader(0) #load data
ind_0,count_0,M200_0,R200_0,group_pos_0,part_pos_0,part_vel_0,part_mass_0=loader(0.1)
ind_1,count_1,M200_1,R200_1,group_pos_1,part_pos_1,part_vel_1,part_mass_1=loader(1)


#converting masses
M200*=10**10; M200_0*=10**10; M200_1*=10**10
part_mass*=10**10

#stack halos
stack_id=np.array([np.argmin(np.abs(M200-10**14))])

pos,vel,m=stacker(stack_id,part_pos,part_vel,ind,M200,R200,group_pos,count,part_mass)
pos_0,vel_0,m_0=stacker(stack_id,part_pos_0,part_vel_0,ind_0,M200_0,R200_0,group_pos_0,count_0,part_mass)
pos_1,vel_1,m_1=stacker(stack_id,part_pos_1,part_vel_1,ind_1,M200_1,R200_1,group_pos_1,count_1,part_mass)

pos_s,vel_s=cartesian_to_spherical(pos,vel)
pos_s_0,vel_s_0=cartesian_to_spherical(pos_0,vel_0)
pos_s_1,vel_s_1=cartesian_to_spherical(pos_1,vel_1)

bin_centret_h=np.logspace(np.log10(0.004),np.log10(1),500); r=bin_centret_h
rho_gausst,vel_gausst,bin_centret=phase_gaussian_kernel(pos_s[:,0],vel_s,1,bin_centret_h,m,var_mass=True)
rho_gausst_0,vel_gausst_0,bin_centret_0=phase_gaussian_kernel(pos_s_0[:,0],vel_s_0,1,bin_centret_h,m_0,var_mass=True)
rho_gausst_1,vel_gausst_1,bin_centret_1=phase_gaussian_kernel(pos_s_1[:,0],vel_s_1,1,bin_centret_h,m_1,var_mass=True)

m_less=mass_prof(pos_s[:,0],m,bin_centret_h)
m_less_0=mass_prof(pos_s_0[:,0],m_0,bin_centret_h)
m_less_1=mass_prof(pos_s_1[:,0],m_1,bin_centret_h)

M2=M200[stack_id[0]]
R2=R200[stack_id[0]]
g_sm=0.004/R2
plt.figure()
ax=plt.subplot(111)
ax.plot(r,rho_gausst,color='#1f77b4',lw=2.0,label='0')
ax.plot(r,rho_gausst_0,color='#ff7f0e',lw=2.0,label='0.1')
ax.plot(r,rho_gausst_1,color='#2ca02c',lw=2.0,label='1')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('R',fontsize=16)
ax.set_ylabel('$\\rho$',fontsize=16)
plt.legend()
ymin=10**-1
ymax=10**2.3
ax.set_ylim(ymin,ymax)
ax.set_xlim(np.min(r),np.max(r))
ax.plot(np.array([g_sm,g_sm]),np.array([ymin,ymax]),color='gray',alpha=0.6)
ax.text(0.05,0.05,'M$_{\\rm{200c}}$=%.2e $h^{-1}M_{sun}$'%(M2),transform=ax.transAxes,fontsize=14)

plt.figure()
ax=plt.subplot(111)
ax.plot(r,vel_gausst[:,0],color='#1f77b4',lw=2.0,label='0')
ax.plot(r,vel_gausst_0[:,0],color='#ff7f0e',lw=2.0,label='0.1')
ax.plot(r,vel_gausst_1[:,0],color='#2ca02c',lw=2.0,label='1')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('R',fontsize=16)
ax.set_ylabel('$\\sigma_{r}$',fontsize=16)
plt.legend(loc=4)
ymin=10**-1
ymax=10**0
ax.set_ylim(ymin,ymax)
ax.set_xlim(np.min(r),np.max(r))
ax.plot(np.array([g_sm,g_sm]),np.array([ymin,ymax]),color='gray',alpha=0.6)
ax.text(0.05,0.05,'M$_{\\rm{200c}}$=%.2e $h^{-1}M_{sun}$'%(M2),transform=ax.transAxes,fontsize=14)

plt.figure()
ax=plt.subplot(111)
ax.plot(r,rho_gausst/vel_gausst[:,0]**3,color='#1f77b4',lw=2.0,label='0')
ax.plot(r,rho_gausst_0/vel_gausst_0[:,0]**3,color='#ff7f0e',lw=2.0,label='0.1')
ax.plot(r,rho_gausst_1/vel_gausst_1[:,0]**3,color='#2ca02c',lw=2.0,label='1')
A=(rho_gausst/vel_gausst[:,0]**3 )[len(r)-50]/(r[len(r)-50])**(-1.875)
ax.plot(r,A*r**(-1.875),'k--')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('R',fontsize=16)
ax.set_ylabel('$\\sigma/\\rho^{3}$',fontsize=16)
plt.legend()
ymin=10**-1
ymax=10**4
ax.set_ylim(ymin,ymax)
ax.set_xlim(np.min(r),np.max(r))
ax.plot(np.array([g_sm,g_sm]),np.array([ymin,ymax]),color='gray',alpha=0.6)
ax.text(0.05,0.05,'M$_{\\rm{200c}}$=%.2e $h^{-1}M_{sun}$'%(M2),transform=ax.transAxes,fontsize=14)

LHS=derivative(r,rho_gausst*vel_gausst[:,0]**2)
RHS=-m_less/r**2*rho_gausst
LHS_0=derivative(r,rho_gausst_0*vel_gausst_0[:,0]**2)
RHS_0=-m_less_0/r**2*rho_gausst_0
LHS_1=derivative(r,rho_gausst_1*vel_gausst_1[:,0]**2)
RHS_1=-m_less_1/r**2*rho_gausst_1

plt.figure()
ax=plt.subplot(111)
ax.plot(r,LHS/RHS,color='#1f77b4',lw=2.0,label='0')
ax.plot(r,LHS_0/RHS_0,color='#ff7f0e',lw=2.0,label='0.1')
ax.plot(r,LHS_1/RHS_1,color='#2ca02c',lw=2.0,label='1')
ax.set_xscale('log')
ax.set_xlabel('R',fontsize=16)
ax.set_ylabel('LHS/RHS',fontsize=16)
plt.legend()
ymin=-3
ymax=5
ax.set_ylim(ymin,ymax)
ax.set_xlim(np.min(r),np.max(r))
ax.plot(np.array([g_sm,g_sm]),np.array([ymin,ymax]),color='gray',alpha=0.6)
ax.text(0.05,0.05,'M$_{\\rm{200c}}$=%.2e $h^{-1}M_{sun}$'%(M2),transform=ax.transAxes,fontsize=14)

plt.figure()
ax=plt.subplot(111)
ax.plot(r,1-vel_gausst[:,1]**2/vel_gausst[:,0]**2,color='#1f77b4',lw=2.0,label='0 $\\beta_{\\theta}$')
ax.plot(r,1-vel_gausst_0[:,1]**2/vel_gausst_0[:,0]**2,color='#ff7f0e',lw=2.0,label='0.1 $\\beta_{\\theta}$')
ax.plot(r,1-vel_gausst_1[:,1]**2/vel_gausst_1[:,0]**2,color='#2ca02c',lw=2.0,label='1 $\\beta_{\\theta}$')
ax.plot(r,1-vel_gausst[:,2]**2/vel_gausst[:,0]**2,color='#1f77b4',linestyle=':',lw=2.0,label='0 $\\beta_{\\phi}$')
ax.plot(r,1-vel_gausst_0[:,2]**2/vel_gausst_0[:,0]**2,color='#ff7f0e',linestyle=':',lw=2.0,label='0.1 $\\beta_{\\phi}$')
ax.plot(r,1-vel_gausst_1[:,2]**2/vel_gausst_1[:,0]**2,color='#2ca02c',linestyle=':',lw=2.0,label='1 $\\beta_{\\phi}$')
ax.set_xscale('log')
ax.set_xlabel('R',fontsize=16)
ax.set_ylabel('$\\beta$',fontsize=16)
plt.legend(loc=4)
ymin=-2
ymax=2
ax.set_ylim(ymin,ymax)
ax.set_xlim(np.min(r),np.max(r))
ax.plot(np.array([g_sm,g_sm]),np.array([ymin,ymax]),color='gray',alpha=0.6)
ax.text(0.05,0.05,'M$_{\\rm{200c}}$=%.2e $h^{-1}M_{sun}$'%(M2),transform=ax.transAxes,fontsize=14)

plt.show()