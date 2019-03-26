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


def loader(loc,snap):

	#loading in particle data for each group
	part_mass=E.readAttribute("PARTDATA",loc,'%03d'%snap,"/Header/MassTable")[1]
	group_id=E.readArray("PARTDATA",loc,'%03d'%snap,"/PartType1/GroupNumber",noH=False)
	sub_group_id=E.readArray("PARTDATA",loc,'%03d'%snap,"/PartType1/SubGroupNumber",noH=False)
	part_pos=E.readArray("PARTDATA",loc,'%03d'%snap,"/PartType1/Coordinates",noH=False)
	part_vel=E.readArray("PARTDATA",loc,'%03d'%snap,"/PartType1/Velocity",noH=False)

	group_id=np.abs(group_id)-1
	sort=np.argsort(group_id)
	group_id=group_id[sort]
	part_pos=part_pos[sort]
	part_vel=part_vel[sort]
	sub_group_id=sub_group_id[sort]
	sub,ind,count=np.unique(group_id,return_index=True,return_counts=True)

	#loading in group properties
	M_200=E.readArray("SUBFIND_GROUP",loc,'%03d'%snap,"FOF/Group_M_Crit200",noH=False)
	R_200=E.readArray("SUBFIND_GROUP",loc,'%03d'%snap,"FOF/Group_R_Crit200",noH=False)
	group_pos=E.readArray("SUBFIND_GROUP",loc,'%03d'%snap,"FOF/GroupCentreOfPotential",noH=False)

	return(ind,count,M_200,R_200,group_pos,part_pos,part_vel,part_mass,sub_group_id)


loc='/hpcdata4/arisbrow/simulations/L100N256_WMAP9/DMONLY_SIDM0.0/data'
tag=34
ind,count,M_200,R_200,group_pos,part_pos,part_vel,part_mass,sub_group_id=loader(loc,tag)


group_num=0

pos=part_pos[ind[group_num]:ind[group_num]+count[group_num]]
uniq=np.unique(sub_group_id[ind[group_num]:ind[group_num]+count[group_num]])

plt.figure()
for i in range(len(uniq)):
	cut=(sub_group_id[ind[group_num]:ind[group_num]+count[group_num]]==uniq[i])
	plt.plot(pos[:,0][cut],pos[:,1][cut],'.',markersize=0.5)

plt.axis('equal')
plt.figure()
cut=(sub_group_id[ind[group_num]:ind[group_num]+count[group_num]]==0)
plt.plot(pos[:,0][cut],pos[:,1][cut],'.',markersize=0.5)
plt.axis('equal')
plt.show()