import numpy as np
import h5py as h5
import eagle3 as E
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
import time
import functions as fn



def pow_l(x,a,b):
	return(np.log(a*(x)**(b)))

def pow(x,a,b,):
	return(a*(x)**(b))

def num_batch(num_p,batch_n):
	ids=[]
	k=0
	while k < len(num_p):
		countt=0
		idds=np.array([k])
		countt+=num_p[k]
		k+=1
		
		while (k< len(num_p) and countt<batch_n):
			idds=np.append(idds,k)
			countt+=num_p[k]
			k+=1
		ids.append(idds)
	return(ids)

def mp_actions(ids):
	nbins=30

	for i in range(len(ids)):
		
		pos,vel,masss,conv_r=fn.stacker(stack_ids[ids[i]],part_pos,part_vel,ind,M200,R200,group_pos,count,part_mass,grav_smoothing,box=box)	
		pos,vel=fn.cartesian_to_spherical(pos,vel)
		
		bin_centre=np.logspace(np.log10(conv_r),np.log10(1),nbins) #logspaced bins to sample from
		rho_gausst,vel_gausst,bin_centret=fn.phase_gaussian_kernel(pos[:,0],vel,1,bin_centre,masss)

		params,err=curve_fit(pow_l,bin_centre,np.log((vel_gausst[:,0]**3/rho_gausst*len(stack_ids[ids[i]]))**(2/3)))
		par[ids[i],:]=params
		profile[ids[i],:]=(vel_gausst[:,0]**3/rho_gausst*len(stack_ids[ids[i]]))**(2/3)
		density[ids[i],:]=rho_gausst/len(stack_ids[ids[i]])
		vel_disp[ids[i],:]=vel_gausst[:,0]
		radius[ids[i],:]=bin_centre

	

cross_sections=['CDM_low','CDM_high','CDM_neg_run','CDM_pos_run']
mass_cut=np.array([10**12,10**9,10**9,10**9])
snaps=np.array([34,33,33,33])
box_size=np.array([100,25,25,25])
num_sims=len(cross_sections)
location=[]
location.append('/hpcdata4/arisbrow/simulations/L100N256_WMAP9/DMONLY_SIDM0.0/data')
location.append('/hpcdata4/sam/PhD/Investigating_Running/RUNS/DMONLY/L025N1024/run_0/data')
location.append('/hpcdata4/sam/PhD/Investigating_Running/RUNS/DMONLY/L025N1024/run_m0p02473/data')
location.append('/hpcdata4/sam/PhD/Investigating_Running/RUNS/DMONLY/L025N1024/run_p0p00791/data')

for j in range(4):
	
	start=time.time()
	print('Loading data')
	ind,count,M200,R200,group_pos,part_pos,part_vel,part_mass=fn.loader(location[j],snaps[j]) #load data
	print('Read data')
	#sorting arrays to be in order of M200
	sort=np.argsort(M200)
	ind=ind[sort]; count=count[sort]; M200=M200[sort]; group_pos=group_pos[sort]; R200=R200[sort];

	#converting masses
	M200*=10**10
	part_mass*=10**10

	nbins=30

	#calculating how to stack
	grav_smoothing=0.004
	stack_ids,num,mass,rad,num_p=fn.halo_to_stack(M200,R200,part_mass,num_stack=1)	#setting to one gaurantess no stacking
	
	cut=np.where(mass>mass_cut[j])[0] #cutting to only consider halos with a mass greater than 10**12 h^-1 M_sun
	stack_idss=[]
	for i in range(len(cut)):
		stack_idss.append(stack_ids[cut[i]])
	stack_ids=stack_idss
	num=num[cut]; mass=mass[cut]; rad=rad[cut]; num_p=num_p[cut]

	par=np.empty((len(mass),2))
	radius=np.empty((len(mass),nbins))
	profile=np.empty((len(mass),nbins))
	density=np.empty((len(mass),nbins))
	vel_disp=np.empty((len(mass),nbins))

	
	batch_n = np.array([1.4*10**5,10**16]) #number of particles (summed N200) to be processed in each bath of the multiprocessing
	ids=num_batch(num_p,batch_n[0])
	ids=list(reversed(ids))
	
	box=box_size[j]
	print('Box size is: '+str(box)+' h^-1Mpc')
	pool=Pool(processes=8)
	pool.map(mp_actions,ids)
	pool.close()
	print('Finished analysing, moving on to next simulation')

	np.save('Entropy_CDM_mass_'+cross_sections[j]+'.npy',mass)
	np.save('Entropy_CDM_fit_params_'+cross_sections[j]+'.npy',par)
	np.save('Entropy_CDM_profile_'+cross_sections[j]+'.npy',profile)
	np.save('Entropy_CDM_density_'+cross_sections[j]+'.npy',density)
	np.save('Entropy_CDM_vel_disp_'+cross_sections[j]+'.npy',vel_disp)
	np.save('Rad_CDM_profile_'+cross_sections[j]+'.npy',radius)

	ind=count=M200=R200=group_pos=part_pos=part_vel=part_mass=None #clear variable from memory

	end=time.time()
	print(end-start)
	
	