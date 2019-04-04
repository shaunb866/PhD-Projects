import numpy as np
cimport numpy as np
cimport libc.math as mt
from libc.stdlib cimport malloc, free

DTYPE = np.float
ctypedef np.float_t DTYPE_t

import cython
@cython.cdivision(True)

cdef int argmin(double *dist):
	cdef int i, ind
	ind=0
	cdef double dist_val=dist[ind]
	while dist_val<0:
		ind+=1
		dist_val=dist[ind]
	return(ind)

def dens_calc(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=2] vel, DTYPE_t m,np.ndarray[DTYPE_t, ndim=1] bins,int nparts, int nparts_vel):
	#calculates density and velocity dispersion using a tricube weight function
	#sort=np.argsort(x)
	#xx=x[sort]
	#x[:]=x[sort] #sorting data
	#vel[:]=vel[sort]

	cdef unsigned int i, j, nparts_0
	cdef int min_ind, min_ind2
	cdef unsigned int len_data=len(x)
	cdef unsigned int len_samp=len(bins)
	cdef unsigned int ndims=vel.shape[1]

	cdef double h, weight_sum, mean, mean_x, mean_y, mean_z, vel_disp_x, vel_disp_y, vel_disp_z, vol_norm, less_h, less_o

	cdef np.ndarray[DTYPE_t, ndim=1] dens=np.empty(len_samp,dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=2] vel_disp=np.empty((len_samp,3),dtype=DTYPE)

	
	cdef double *dist_near = <double *> malloc(nparts * sizeof(double))
	cdef double *diff = <double *> malloc(len_data * sizeof(double))
	cdef double *dist = <double *> malloc(len_data * sizeof(double))
	cdef int *min_ids = <int *> malloc(len_samp * sizeof(int))

	#calculate the density
	for i in range(len_samp):
		

		#resetting relevent variables
		nparts_0=nparts
		weight_sum=0.0
		
		#calcualting distance
		for j in range(len_data):
			diff[j]=x[j]-bins[i]
			dist[j]=mt.fabs(diff[j])
		
		min_ind=argmin(diff)
		min_ids[i]=min_ind
		if (min_ind-nparts_0/2)<0:
			nparts_0=2*min_ind
		if (min_ind+nparts_0/2)>=len_data:
			nparts_0=2*(len_data-min_ind)

		
		for j in range(nparts_0):
			dist_near[j]=dist[min_ind-nparts_0/2+j]
		h=dist_near[0]
		
		for j in range(nparts_0):
			dist_near[j]=dist_near[j]/h

		
		for j in range(nparts_0):
			less_h=int(dist_near[j]/0.5)
			if less_h==0:
				weight_sum+=(1-6.0*dist_near[j]**2+6.0*dist_near[j]**3)
			else:
				weight_sum+=(2*(1-dist_near[j])**3)

		vol_norm=np.pi*h*(0.25*h**2+3*bins[i]**2)

		dens[i]=m*weight_sum/(vol_norm)
		
		

	cdef double *dist_near2 = <double *> malloc(nparts_vel * sizeof(double))
	cdef double *vel_x = <double *> malloc(nparts_vel * sizeof(double))
	cdef double *vel_y = <double *> malloc(nparts_vel * sizeof(double))
	cdef double *vel_z = <double *> malloc(nparts_vel * sizeof(double))
	cdef double *weight = <double *> malloc(nparts_vel * sizeof(double))

	#calculating the velocity dispersion in each direction
	for i in range(len_samp):
		

		#resetting relevent variables
		nparts_0=nparts_vel
		weight_sum=0.0
		vel_disp_x=0.0
		vel_disp_y=0.0
		vel_disp_z=0.0
		#calcualting distance
		for j in range(len_data):
			dist[j]=mt.fabs(x[j]-bins[i])
		
		min_ind=min_ids[i]

		if (min_ind-nparts_0/2)<0:
			nparts_0=2*min_ind
		if (min_ind+nparts_0/2)>=len_data:
			nparts_0=2*(len_data-min_ind)

		
		for j in range(nparts_0):
			dist_near2[j]=dist[min_ind-nparts_0/2+j]
			vel_x[j]=vel[min_ind-nparts_0/2+j,0]
			vel_y[j]=vel[min_ind-nparts_0/2+j,1]
			vel_z[j]=vel[min_ind-nparts_0/2+j,2]
		h=dist_near2[0]
		
		for j in range(nparts_0):
			dist_near2[j]=dist_near2[j]/h
			

		for j in range(nparts_0):
			less_h=int(dist_near2[j]/0.5)
			if less_h==0:
				weight[j]=(1-6.0*dist_near2[j]**2+6.0*dist_near2[j]**3)
			else:
				weight[j]=(2*(1-dist_near2[j])**3)

		for j in range(nparts_0):
			weight_sum+=weight[j]
			
		for j in range(nparts_0):
			mean_x+=weight[j]*vel_x[j]/weight_sum
			mean_y+=weight[j]*vel_y[j]/weight_sum
			mean_z+=weight[j]*vel_z[j]/weight_sum

		

		for j in range(nparts_0):
			vel_disp_x+=weight[j]*(vel_x[j]-mean_x)**2/weight_sum
			vel_disp_y+=weight[j]*(vel_y[j]-mean_y)**2/weight_sum
			vel_disp_z+=weight[j]*(vel_z[j]-mean_z)**2/weight_sum

		vel_disp[i,0]=(vel_disp_x)**0.5
		vel_disp[i,1]=(vel_disp_y)**0.5
		vel_disp[i,2]=(vel_disp_z)**0.5
	
	return(dens,vel_disp)

	