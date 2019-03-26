import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

def var_func(vel):
	var=np.var(vel)
	return(var)
def loader(loc):

	#loading in particle data for each group
	
	particle_sub=h5.File(loc+'/data/particledata_034/eagle_subfind_particles_034.0.hdf5')
	
	
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
	h=h5.File(loc+'/data/groups_034/eagle_subfind_tab_034.0.hdf5')
	
	M_200=h['FOF']['Group_M_Crit200'][()]
	R_200=h['FOF']['Group_R_Crit200'][()]
	group_pos=h['FOF']['GroupCentreOfPotential'][()]
	for i in range(31):
		h=h5.File(loc+'/data/groups_034/eagle_subfind_tab_034.'+str(i+1)+'.hdf5')
		
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
	
	bin_edges=np.logspace(np.log10(0.004),np.log10(4*R200),nbin)
        
	
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

def phase_gaussian_kernel(r,vel,R200,bin_centre,m):
	#m must be an array for the mass of each particle, even if this is just a constant
	num_p=int(59.1*np.sum(r<R200)**(0.32))
	num_v=int(2.9*np.sum(r<R200)**(0.8))
	vel=vel[np.where(r<5.0*R200)]
	m=m[np.where(r<5.0*R200)]
	r=r[np.where(r<5.0*R200)]

	#num_p=int(3.146*np.sum(r<R200)**(0.79))
	rho=np.empty(len(bin_centre))
	vel_dd=np.empty((len(bin_centre),3))
	
	for i in range(len(bin_centre)):
		dist=np.abs(r-bin_centre[i])
		sort=np.argsort(dist)
		dist_sort=dist[sort]
		vel_sort=vel[sort]
		m_sort=m[sort]
		h=dist_sort[num_p-1]
		if h>bin_centre[i]:
			h=bin_centre[i]
			nnn=np.sum(dist<h)
			
			if nnn==0:
				h=dist_sort[2]
				
		
		
		dist_half=dist_sort[dist_sort/h<=0.5]/h
		dist_one=dist_sort[(dist_sort/h>=0.5) & (dist_sort/h<=1)]/h
		vel_half=vel_sort[dist_sort/h<=0.5]
		vel_one=vel_sort[(dist_sort/h>=0.5) & (dist_sort/h<=1)]
		
		m_half=m_sort[dist_sort/h<=0.5]
		m_one=m_sort[(dist_sort/h>=0.5) & (dist_sort/h<=1)]

		
		vol_norm=np.pi*h*(0.25*h**2+3*bin_centre[i]**2)
		weight1=(1-6*dist_half**2+6*dist_half**3)
		weight2=2*(1-dist_one)**3
		
		dmdr=(np.sum(weight1*m_half)+np.sum(weight2*m_one))
		rho[i]=dmdr/(vol_norm)
	
	for i in range(len(bin_centre)):
		dist=np.abs(r-bin_centre[i])
		sort=np.argsort(dist)
		dist_sort=dist[sort]
		vel_sort=vel[sort]
		m_sort=m[sort]
		h=dist_sort[num_v-1]
		if h>bin_centre[i]:
			h=bin_centre[i]
			nnn=np.sum(dist<h)
			if nnn==0:
				h=dist_sort[1]
				
		
		
		dist_half=dist_sort[dist_sort/h<=0.5]/h
		dist_one=dist_sort[(dist_sort/h>=0.5) & (dist_sort/h<=1)]/h
		vel_half=vel_sort[dist_sort/h<=0.5]
		vel_one=vel_sort[(dist_sort/h>=0.5) & (dist_sort/h<=1)]
		
		m_half=m_sort[dist_sort/h<=0.5]
		m_one=m_sort[(dist_sort/h>=0.5) & (dist_sort/h<=1)]

		
		vol_norm=np.pi*h*(0.25*h**2+3*bin_centre[i]**2)
		weight1=(1-6*dist_half**2+6*dist_half**3)
		weight2=2*(1-dist_one)**3
		weight_sum=np.sum(weight1)+np.sum(weight2)
		
		vel_meanx=(np.sum(weight1*vel_half[:,0])+np.sum(weight2*vel_one[:,0]))/weight_sum
		vel_meany=(np.sum(weight1*vel_half[:,1])+np.sum(weight2*vel_one[:,1]))/weight_sum
		vel_meanz=(np.sum(weight1*vel_half[:,2])+np.sum(weight2*vel_one[:,2]))/weight_sum

		vel_dispx=(np.sum(weight1*(vel_half[:,0]-vel_meanx)**2)+np.sum(weight2*(vel_one[:,0]-vel_meanx)**2))/weight_sum
		vel_dispy=(np.sum(weight1*(vel_half[:,1]-vel_meany)**2)+np.sum(weight2*(vel_one[:,1]-vel_meany)**2))/weight_sum
		vel_dispz=(np.sum(weight1*(vel_half[:,2]-vel_meanz)**2)+np.sum(weight2*(vel_one[:,2]-vel_meanz)**2))/weight_sum

		vel_dd[i,0]=vel_dispx**0.5
		vel_dd[i,1]=vel_dispy**0.5
		vel_dd[i,2]=vel_dispz**0.5
		
	return(rho,vel_dd)

def cartesian_to_spherical(pos,vel):
	pos[np.where(pos==0.0)]=0.0001
	pos_spherical=np.empty(pos.shape)
	pos_spherical[:,0]=np.sqrt(pos[:,0]**2+pos[:,1]**2+pos[:,2]**2)
	pos_spherical[:,1]=np.arccos(pos[:,2]/pos_spherical[:,0])
	pos_spherical[:,2]=np.arctan(pos[:,1]/pos[:,0])

	vel_spherical=np.empty(pos.shape)
	vel_spherical[:,0]=vel[:,0]*np.sin(pos_spherical[:,1])*np.cos(pos_spherical[:,2])+vel[:,1]*np.sin(pos_spherical[:,1])*np.sin(pos_spherical[:,2])+vel[:,2]*np.cos(pos_spherical[:,1])
	vel_spherical[:,1]=vel[:,0]*np.cos(pos_spherical[:,1])*np.cos(pos_spherical[:,2])+vel[:,1]*np.cos(pos_spherical[:,1])*np.sin(pos_spherical[:,2])-vel[:,2]*np.sin(pos_spherical[:,1])
	vel_spherical[:,2]=-vel[:,0]*np.sin(pos_spherical[:,2])+vel[:,1]*np.cos(pos_spherical[:,2])
	return(pos_spherical,vel_spherical)

def stacker(stack_id,part_pos,part_vel,ind,M200,R200,group_pos,count,part_mass,box=100):
	g=4.3009*10**(-3)
	total_num=np.sum(count[stack_id])
	pos=np.empty((total_num,3))
	vel=np.empty((total_num,3))
	mass=np.empty(total_num)
	counter=0
	conv_rad=np.empty(len(stack_id))
	for i in range(len(stack_id)):
		group_num=stack_id[i]
		poss=part_pos[ind[group_num]:ind[group_num]+count[group_num]]
		vels=part_vel[ind[group_num]:ind[group_num]+count[group_num]]
		m2=M200[group_num]
		r2=R200[group_num]
		poss[:,0]-=group_pos[group_num,0]; poss[:,1]-=group_pos[group_num,1]; poss[:,2]-=group_pos[group_num,2]
		poss-=(poss/(box/2)).astype(np.int)*box #deal with periodic boundary condition
		r=np.sqrt(poss[:,0]**2+poss[:,1]**2+poss[:,2]**2)
		r_less=np.where(r<r2)
		rr_less=np.sort(r[r_less])

		#calculating convergence radius in r/R200
		n_less=(np.arange(len(rr_less))+1)
		m_less=n_less*part_mass
		rho_rat=m_less[1:]/(4/3*np.pi*rr_less[1:]**3)*8*np.pi*g/(3*100**2)*10**(-6)
		RHS=200**0.5/8*n_less[1:]/np.log(n_less[1:])*rho_rat**(-0.5)
		conv_rad[i]=rr_less[np.nanargmin(np.abs(RHS-0.6))]/r2
		
		vel_x=np.mean(vels[:,0][r_less]); vel_y=np.mean(vels[:,1][r_less]); vel_z=np.mean(vels[:,2][r_less])
		vels[:,0]-=vel_x; vels[:,1]-=vel_y; vels[:,2]-=vel_z

		v_circ2=6.557*10**(-5)*np.sqrt(m2/r2)
		pos[counter:counter+count[group_num],:]=poss/r2
		vel[counter:counter+count[group_num],:]=vels/v_circ2
		mass[counter:counter+count[group_num]]=np.ones(count[group_num])*part_mass/m2
		counter+=count[group_num]
	
	

	return(pos,vel,mass,np.max(conv_rad))

def mass_prof(r,m,bin_centre):
	mas_r=np.empty(len(bin_centre))
	for i in range(len(bin_centre)):
		mas_r[i]=np.sum(m[r<bin_centre[i]])
	return(mas_r)

s_length=0.004 #gravitational softening length of the sim in h^-1 Mpc




cross_sections=np.array([0.0,0.1,1.0,5.0,10.0])
num_sims=len(cross_sections)
location=[]
for i in range(num_sims):
	location.append('/hpcdata4/arisbrow/simulations/L100N256_WMAP9/DMONLY_SIDM%.1f' % cross_sections[i])



#stack halos
mass_range=np.array([10*12,10**(12.5),10**(13),10**(13.5),10**(14),10**(15)])
label=np.array([12,12.5,13,13.5,14,15])
for j in range(num_sims):
	print(j)
	ind,count,M200,R200,group_pos,part_pos,part_vel,part_mass=loader(location[j])
	M200*=10**10
	part_mass*=10**10
	for i in range(len(mass_range)-1):

		m_min=mass_range[i]
		m_max=mass_range[i+1]
		stack_id=np.where((M200<m_max) & (M200>m_min))[0]

		pos,vel,m,conv_rad=stacker(stack_id,part_pos,part_vel,ind,M200,R200,group_pos,count,part_mass)

		pos_s,vel_s=cartesian_to_spherical(pos,vel)

		rho_histt,vel_disp_histt,bin_centret=phase_bin_log(pos_s[:,0],vel_s,1,30,m,var_mass=True)
		rho_gausst,vel_gausst=phase_gaussian_kernel(pos_s[:,0],vel_s,1,bin_centret,m)

		m_less=mass_prof(pos_s[:,0],m,bin_centret)

		np.save('rho_gauss_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]),rho_gausst)
		np.save('rho_squar_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]),rho_histt)
		

		np.save('sig_gauss_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]),vel_gausst)
		np.save('sig_squar_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]),vel_disp_histt)
	
		
		np.save('r_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]),bin_centret)
		np.save('m_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]),m_less)
		np.save('nhalo_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]),len(stack_id))
		np.save('r_conv_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]),conv_rad)
		print(i)




