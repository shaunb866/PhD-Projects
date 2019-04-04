import numpy as np
import matplotlib.pyplot as plt
import eagle3 as E


loc=[]
loc.append('/hpcdata4/arisbrow/simulations/L100N256_WMAP9/DMONLY_SIDM0.0/data')
tag=np.array([34,28,24,20])
nsnaps=35
redshift=np.array([])
a=np.array([])
dens_phy=np.empty(nsnaps)
dens_com=np.empty(nsnaps)
vel_disp_phy=np.empty(nsnaps)
vel_disp_com=np.empty(nsnaps)
plt.figure()

for i in range(nsnaps):
	print(i)
	z=E.readAttribute("PARTDATA",loc[0],'%03d'%(i),"/Header/Redshift")
	redshift=np.append(redshift,z)
	aa=E.readAttribute("PARTDATA",loc[0],'%03d'%(i),"/Header/ExpansionFactor")
	a=np.append(a,aa)
	m=E.readAttribute("PARTDATA",loc[0],'%03d'%(i),"/Header/MassTable")[1]
	box=E.readAttribute("PARTDATA",loc[0],'%03d'%(i),"/Header/BoxSize")
	vel=E.readArray("SNAPSHOT",loc[0],'%03d'%(i),"/PartType1/Velocity",noH=False,physicalUnits=False)
	N=len(vel) #number of particles in the simulation
	dens_com[i]=N*m/box**3
	dens_phy[i]=N*m/(aa*box)**3

	vel_disp_com[i]=(np.var(vel[:,0]*aa**(-1.5))+np.var(vel[:,1]*aa**(-1.5))+np.var(vel[:,2]*aa**(-1.5)))**0.5/3**0.5
	vel_disp_phy[i]=(np.var(vel[:,0]*aa**0.5)+np.var(vel[:,1]*aa**0.5)+np.var(vel[:,2]*aa**0.5))**0.5/3**0.5
	plt.hist(vel.flatten(),bins=100,normed=True,histtype='step')

plt.figure()
plt.plot(redshift+1,vel_disp_com,label='comoving')
plt.plot(redshift+1,vel_disp_phy,label='physical')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('z+1')
plt.ylabel('vel_disp')
plt.legend()

plt.figure()
plt.plot(redshift+1,dens_com,label='comoving')
plt.plot(redshift+1,dens_phy,label='physical')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('z+1')
plt.ylabel('density')
plt.legend()


plt.figure()
plt.plot(redshift+1,vel_disp_com**2/dens_com**(2/3),label='comoving')
plt.plot(redshift+1,vel_disp_phy**2/dens_phy**(2/3),label='physical')
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('z+1')
plt.ylabel('entropy')
plt.legend()

plt.figure()
plt.plot(a,vel_disp_com,label='comoving')
plt.plot(a,vel_disp_phy,label='physical')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('a')
plt.ylabel('vel_disp')
plt.legend()

plt.figure()
plt.plot(a,vel_disp_com**2/dens_com**(2/3),label='comoving')
plt.plot(a,vel_disp_phy**2/dens_phy**(2/3),label='physical')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('a')
plt.ylabel('entropy')
plt.legend()

plt.show()