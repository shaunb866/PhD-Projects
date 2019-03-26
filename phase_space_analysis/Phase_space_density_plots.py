import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def derivative(x,y):
    dydx=np.empty(len(x))

    for i in range(len(x)-2):
        h2=x[i+1]-x[i]
        h1=x[i+2]-x[i+1]
        dydx[i+1]=-h1/(h2*(h1+h2))*y[i]+(h1-h2)/(h1*h2)*y[i+1]+h2/(h1*(h1+h2))*y[i+2]
        
    dydx[0]=(y[1]-y[0])/(x[1]-x[0])
    dydx[len(x)-1]=(y[len(x)-1]-y[len(x)-2])/(x[len(x)-1]-x[len(x)-2])
    return(dydx)

nsims=5
low_len=29
high_len=200
mass_range=np.array([10**12,10**(12.5),10**(13),10**(13.5),10**(14),10**(14.5)])
label=np.array([12,12.5,13,13.5,14,15])
cross_sections=np.array([0.0,0.1,1.0,5.0,10.0])


rho_gauss=np.empty((nsims,len(mass_range)-1,low_len))
rho_hist=np.empty((nsims,len(mass_range)-1,low_len))
vel_gauss=np.empty((nsims,len(mass_range)-1,low_len,3))
vel_hist=np.empty((nsims,len(mass_range)-1,low_len,3))
num_halo=np.empty((nsims,len(mass_range)-1))
m_less=np.empty((nsims,len(mass_range)-1,low_len))
r_conv=np.empty((nsims,len(mass_range)-1))
r=np.load('r_0_12.0_12.5.npy')


for i in range(nsims):
	for j in range(len(label)-1):
		rho_gauss[i,j,:]=np.load('rho_gauss_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]))
		rho_hist[i,j,:]=np.load('rho_squar_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]))
		vel_gauss[i,j,:,:]=np.load('sig_gauss_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]))
		vel_hist[i,j,:,:]=np.load('sig_squar_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]))

		num_halo[i,j]=np.load('nhalo_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]))
		m_less[i,j,:]=np.load('m_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]))

		r_conv[i,j]=np.load('r_conv_%.1f_%.1f_%.1f.npy'%(cross_sections[j],label[i],label[i+1]))

m_av=np.empty(len(mass_range)-1)
m_avv=np.empty(len(mass_range)-1)
for i in range(len(m_av)):
	m_av[i]=mass_range[i]
	m_avv[i]=(mass_range[i+1]+mass_range[i])/2

grav_smooth=0.004
r_sm=grav_smooth/(4.30*10**(-15)*m_av)**(1/3)*2.41 #times by 2.41 as this is where the true grav force dominates over the artifical softening length
r_sm_av=grav_smooth/(4.30*10**(-15)*m_avv)**(1/3)
#plot radial profiles
fig=plt.figure()
gs=gridspec.GridSpec(2,3)
gs.update(wspace=0.0, hspace=0.0)

ax=[]
for i in range(5):
	ax.append(plt.subplot(gs[i]))

Rs=np.empty(5)
for i in range(5):
	ax[i].plot(r,rho_hist[0,i,:]/num_halo[0,i],color='#1f77b4',linestyle=':',lw=2.0)
	ax[i].plot(r,rho_hist[1,i,:]/num_halo[1,i],color='#ff7f0e',linestyle=':',lw=2.0)
	ax[i].plot(r,rho_hist[2,i,:]/num_halo[2,i],color='#2ca02c',linestyle=':',lw=2.0)
	ax[i].plot(r,rho_gauss[0,i,:]/num_halo[0,i],color='#1f77b4',linestyle='-',lw=2.0)
	ax[i].plot(r,rho_gauss[1,i,:]/num_halo[1,i],color='#ff7f0e',linestyle='-',lw=2.0)
	ax[i].plot(r,rho_gauss[2,i,:]/num_halo[2,i],color='#2ca02c',linestyle='-',lw=2.0)
	
	ax[i].set_xscale('log')
	ax[i].set_yscale('log')
	ax[i].set_xlim(np.min(r),np.max(r))
	ylow=10**(-1)
	yhigh=10**(2.4)
	ax[i].set_ylim(ylow,yhigh)

	ax[i].plot(np.ones(2)*r_conv[0,i],np.array([ylow,yhigh]),'--',color='#1f77b4',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[1,i],np.array([ylow,yhigh]),'--',color='#ff7f0e',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[2,i],np.array([ylow,yhigh]),'--',color='#2ca02c',alpha=0.75)
	ax[i].text(0.05,0.05,'log M$_{\\rm{200c}}$='+str(label[i])+' -'+str(label[i+1]),transform=ax[i].transAxes,fontsize=16)

	Rs[i]=r[np.argmax(rho_gauss[0,i,:]*r**2)]
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[4].set_yticklabels([])


ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#1f77b4',label='$\\sigma=0$',lw=2.0)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#ff7f0e',label='$\\sigma=0.1$',lw=2.0)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#2ca02c',label='$\\sigma=1$',lw=2.0)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k-',label='Guassian Kernel',lw=2.0)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k:',label='Histogram',lw=2.0)
ax[0].legend()
ax[0].legend(frameon=False)

fig.text(0.5,0.04,'$R$',fontsize=20,ha='center')
fig.text(0.04,0.5,'$\\rho$',fontsize=20,va='center',rotation='vertical')


Vs=np.empty(5)
#plotting velocity dispersion profiles
fig=plt.figure()
gs=gridspec.GridSpec(2,3)
gs.update(wspace=0.0, hspace=0.0)

ax=[]
for i in range(5):
	ax.append(plt.subplot(gs[i]))
for i in range(5):
	ax[i].plot(r,vel_hist[0,i,:,0],color='#1f77b4',linestyle=':',lw=1.5)
	ax[i].plot(r,np.sqrt(m_less[0,i,:]/num_halo[0,i]/r)/3**0.5,'k--',lw=1.5,alpha=0.5)
	ax[i].plot(r,vel_hist[1,i,:,0],color='#ff7f0e',linestyle=':',lw=1.5)
	ax[i].plot(r,vel_hist[2,i,:,0],color='#2ca02c',linestyle=':',lw=1.5)
	ax[i].plot(r,vel_gauss[0,i,:,0],color='#1f77b4',linestyle='-',lw=1.5)
	ax[i].plot(r,vel_gauss[1,i,:,0],color='#ff7f0e',linestyle='-',lw=1.5)
	ax[i].plot(r,vel_gauss[2,i,:,0],color='#2ca02c',linestyle='-',lw=1.5)

	ax[i].set_xscale('log')
	#ax[i].set_yscale('log')
	ax[i].set_xlim(np.min(r),np.max(r))

	ylow=0.4
	yhigh=1.0
	ax[i].set_ylim(ylow,yhigh)

	ax[i].plot(np.ones(2)*r_conv[0,i],np.array([ylow,yhigh]),'--',color='#1f77b4',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[1,i],np.array([ylow,yhigh]),'--',color='#ff7f0e',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[2,i],np.array([ylow,yhigh]),'--',color='#2ca02c',alpha=0.75)
	ax[i].text(0.05,0.05,'log M$_{\\rm{200c}}$='+str(label[i])+' -'+str(label[i+1]),transform=ax[i].transAxes,fontsize=14)

	Vs[i]=r[np.argmax(vel_gauss[0,i,:,0])]
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[4].set_yticklabels([])

ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#1f77b4',label='$\\sigma=0$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#ff7f0e',label='$\\sigma=0.1$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#2ca02c',label='$\\sigma=1$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k-',label='Guassian Kernel',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k:',label='Histogram',lw=1.5)
ax[0].legend()
ax[0].legend(frameon=False)

fig.text(0.5,0.04,'$R$',fontsize=16,ha='center')
fig.text(0.04,0.5,'$\\sigma$',fontsize=16,va='center',rotation='vertical')


#plotting phase space density
fig=plt.figure()
gs=gridspec.GridSpec(2,3)
gs.update(wspace=0.0, hspace=0.0)

ax=[]
for i in range(5):
	ax.append(plt.subplot(gs[i]))
for i in range(5):
	
	ax[i].plot(r,rho_hist[0,i,:]/vel_hist[0,i,:,0]**3/num_halo[0,i],color='#1f77b4',linestyle=':',lw=1.5)
	ax[i].plot(r,rho_hist[1,i,:]/vel_hist[1,i,:,0]**3/num_halo[0,i],color='#ff7f0e',linestyle=':',lw=1.5)
	ax[i].plot(r,rho_hist[2,i,:]/vel_hist[2,i,:,0]**3/num_halo[0,i],color='#2ca02c',linestyle=':',lw=1.5)
	ax[i].plot(r,rho_gauss[0,i,:]/vel_gauss[0,i,:,0]**3/num_halo[0,i],color='#1f77b4',linestyle='-',lw=1.5)
	ax[i].plot(r,rho_gauss[1,i,:]/vel_gauss[1,i,:,0]**3/num_halo[0,i],color='#ff7f0e',linestyle='-',lw=1.5)
	ax[i].plot(r,rho_gauss[2,i,:]/vel_gauss[2,i,:,0]**3/num_halo[0,i],color='#2ca02c',linestyle='-',lw=1.5)

	A=(rho_gauss[0,i,:]/vel_gauss[0,i,:,0]**3/num_halo[0,i])[len(r)-3]/(r[len(r)-3])**(-1.875)
	ax[i].plot(r,A*r**(-1.875),'k--',lw=1.5)
	ax[i].set_xscale('log')
	ax[i].set_yscale('log')
	ax[i].set_xlim(np.min(r),np.max(r))
	ylow=10**(-1)
	yhigh=10**3.5
	ax[i].set_ylim(ylow,yhigh)

	ax[i].plot(np.ones(2)*r_conv[0,i],np.array([ylow,yhigh]),'--',color='#1f77b4',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[1,i],np.array([ylow,yhigh]),'--',color='#ff7f0e',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[2,i],np.array([ylow,yhigh]),'--',color='#2ca02c',alpha=0.75)
	ax[i].text(0.05,0.05,'log M$_{\\rm{200c}}$='+str(label[i])+' -'+str(label[i+1]),transform=ax[i].transAxes,fontsize=14)

ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[4].set_yticklabels([])

ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#1f77b4',label='$\\sigma=0$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#ff7f0e',label='$\\sigma=0.1$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#2ca02c',label='$\\sigma=1$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k-',label='Guassian Kernel',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k:',label='Histogram',lw=1.5)
ax[0].legend()
ax[0].legend(frameon=False)
fig.text(0.5,0.04,'$R$',fontsize=16,ha='center')
fig.text(0.04,0.5,'$\\rho /\\sigma^3$',fontsize=16,va='center',rotation='vertical')


#calculating and plotting jeans equation physically
LHS_gauss=np.empty(rho_gauss.shape)
RHS_gauss=np.empty(rho_gauss.shape)
LHS_gauss_h=np.empty(rho_gauss_h.shape)
RHS_gauss_h=np.empty(rho_gauss_h.shape)

for i in range(len(rho_hist[:,0,0])):
	for j in range(len(rho_hist[0,:,0])):
		LHS_gauss_h[i,j,:]=derivative(r_h,rho_gauss_h[i,j,:]*vel_gauss_h[i,j,:,0]**2/(num_halo[i,j]))/(rho_gauss_h[i,j,:]/num_halo[i,j])+(2*vel_gauss_h[i,j,:,0]**2-vel_gauss_h[i,j,:,1]**2-vel_gauss_h[i,j,:,2]**2)
		RHS_gauss_h[i,j,:]=-m_less_h[i,j,:]/(r_h**2*num_halo[i,j])
		
		

fig=plt.figure()
gs=gridspec.GridSpec(2,3)
gs.update(wspace=0.0, hspace=0.0)

ax=[]
for i in range(5):
	ax.append(plt.subplot(gs[i]))
for i in range(5):
	ax[i].plot(r_h,LHS_gauss_h[0,i,:]/RHS_gauss_h[0,i,:],color='#1f77b4',linestyle='-',lw=1.5)
	ax[i].plot(r_h,LHS_gauss_h[1,i,:]/RHS_gauss_h[1,i,:],color='#ff7f0e',linestyle='-',lw=1.5)
	ax[i].plot(r_h,LHS_gauss_h[2,i,:]/RHS_gauss_h[2,i,:],color='#2ca02c',linestyle='-',lw=1.5)

	
	ax[i].set_xscale('log')
	ax[i].set_yscale('log')
	ax[i].set_xlim(np.min(r_h),np.max(r_h))
	ylow=10**(-1)
	yhigh=10**1
	ax[i].set_ylim(ylow,yhigh)

	ax[i].plot(np.ones(2)*r_conv[0,i],np.array([ylow,yhigh]),'--',color='#1f77b4',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[1,i],np.array([ylow,yhigh]),'--',color='#ff7f0e',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[2,i],np.array([ylow,yhigh]),'--',color='#2ca02c',alpha=0.75)

	ax[i].text(0.05,0.05,'log M$_{\\rm{200c}}$='+str(label[i])+' -'+str(label[i+1]),transform=ax[i].transAxes,fontsize=14)


ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[4].set_yticklabels([])

ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#1f77b4',label='$\\sigma=0$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#ff7f0e',label='$\\sigma=0.1$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#2ca02c',label='$\\sigma=1$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k-',label='Guassian Kernel',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k:',label='Histogram',lw=1.5)
ax[0].legend()
ax[0].legend(frameon=False)
fig.text(0.5,0.04,'$R$',fontsize=16,ha='center')
fig.text(0.04,0.5,'$\\rho /\\sigma^3$',fontsize=16,va='center',rotation='vertical')



#calculating and plotting jeans equation inculding artificial smoothing length


fig=plt.figure()
gs=gridspec.GridSpec(2,3)
gs.update(wspace=0.0, hspace=0.0)

ax=[]
for i in range(5):
	ax.append(plt.subplot(gs[i]))
for i in range(5):
	ax[i].plot(r_h,1-vel_gauss_h[0,i,:,1]**2/vel_gauss_h[0,i,:,0]**2,color='#1f77b4',linestyle='-',lw=1.5)
	ax[i].plot(r_h,1-vel_gauss_h[1,i,:,1]**2/vel_gauss_h[0,i,:,0]**2,color='#ff7f0e',linestyle='-',lw=1.5)
	ax[i].plot(r_h,1-vel_gauss_h[2,i,:,1]**2/vel_gauss_h[0,i,:,0]**2,color='#2ca02c',linestyle='-',lw=1.5)
	ax[i].plot(r_h,1-vel_gauss_h[0,i,:,2]**2/vel_gauss_h[0,i,:,0]**2,color='#1f77b4',linestyle='--',lw=1.5)
	ax[i].plot(r_h,1-vel_gauss_h[1,i,:,2]**2/vel_gauss_h[0,i,:,0]**2,color='#ff7f0e',linestyle='--',lw=1.5)
	ax[i].plot(r_h,1-vel_gauss_h[2,i,:,2]**2/vel_gauss_h[0,i,:,0]**2,color='#2ca02c',linestyle='--',lw=1.5)
	

	ax[i].set_xscale('log')
	#ax[i].set_yscale('log')
	ax[i].set_xlim(np.min(r_h),np.max(r_h))
	ylow=-0.5
	yhigh=0.5
	ax[i].set_ylim(ylow,yhigh)
	ax[i].plot(r_h,np.ones(len(r_h))*0.0,'k--')
	ax[i].plot(np.ones(2)*r_conv[0,i],np.array([ylow,yhigh]),'--',color='#1f77b4',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[1,i],np.array([ylow,yhigh]),'--',color='#ff7f0e',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[2,i],np.array([ylow,yhigh]),'--',color='#2ca02c',alpha=0.75)
	ax[i].text(0.05,0.05,'log M$_{\\rm{200c}}$='+str(label[i])+' -'+str(label[i+1]),transform=ax[i].transAxes,fontsize=14)

ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[4].set_yticklabels([])

ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#1f77b4',label='$\\sigma=0$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#ff7f0e',label='$\\sigma=0.1$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#2ca02c',label='$\\sigma=1$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k-',label='$\\beta_{\\theta}$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k--',label='$\\beta_{\\phi}$',lw=1.5)

ax[0].legend()
ax[0].legend(frameon=False)
fig.text(0.5,0.04,'$R$',fontsize=16,ha='center')
fig.text(0.04,0.5,'$\\rho /\\sigma^3$',fontsize=16,va='center',rotation='vertical')



#Plotting bulk velocities
fig=plt.figure()
gs=gridspec.GridSpec(2,3)
gs.update(wspace=0.0, hspace=0.0)

ax=[]
for i in range(5):
	ax.append(plt.subplot(gs[i]))
for i in range(5):
	ax[i].plot(r_h,vel_bulk_h[0,i,:,0],color='#1f77b4',linestyle='-',lw=1.5)
	ax[i].plot(r_h,vel_bulk_h[1,i,:,0],color='#ff7f0e',linestyle='-',lw=1.5)
	ax[i].plot(r_h,vel_bulk_h[2,i,:,0],color='#2ca02c',linestyle='-',lw=1.5)
	ax[i].plot(r_h,vel_bulk_h[0,i,:,1],color='#1f77b4',linestyle='--',lw=1.5)
	ax[i].plot(r_h,vel_bulk_h[1,i,:,1],color='#ff7f0e',linestyle='--',lw=1.5)
	ax[i].plot(r_h,vel_bulk_h[2,i,:,1],color='#2ca02c',linestyle='--',lw=1.5)
	ax[i].plot(r_h,vel_bulk_h[0,i,:,2],color='#1f77b4',linestyle=':',lw=1.5)
	ax[i].plot(r_h,vel_bulk_h[1,i,:,2],color='#ff7f0e',linestyle=':',lw=1.5)
	ax[i].plot(r_h,vel_bulk_h[2,i,:,2],color='#2ca02c',linestyle=':',lw=1.5)

	ax[i].set_xscale('log')
	#ax[i].set_yscale('log')
	ax[i].set_xlim(np.min(r_h),np.max(r_h))
	ylow=-0.5
	yhigh=0.5
	ax[i].set_ylim(ylow,yhigh)
	ax[i].plot(r_h,np.ones(len(r_h))*0.0,'k--')
	ax[i].plot(np.ones(2)*r_conv[0,i],np.array([ylow,yhigh]),'--',color='#1f77b4',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[1,i],np.array([ylow,yhigh]),'--',color='#ff7f0e',alpha=0.75)
	ax[i].plot(np.ones(2)*r_conv[2,i],np.array([ylow,yhigh]),'--',color='#2ca02c',alpha=0.75)
	ax[i].text(0.05,0.05,'log M$_{\\rm{200c}}$='+str(label[i])+' -'+str(label[i+1]),transform=ax[i].transAxes,fontsize=14)

ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[4].set_yticklabels([])

ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#1f77b4',label='$\\sigma=0$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#ff7f0e',label='$\\sigma=0.1$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color='#2ca02c',label='$\\sigma=1$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k-',label='$\\beta_{\\theta}$',lw=1.5)
ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k--',label='$\\beta_{\\phi}$',lw=1.5)

ax[0].legend()
ax[0].legend(frameon=False)
fig.text(0.5,0.04,'$R$',fontsize=16,ha='center')
fig.text(0.04,0.5,'$\\rho /\\sigma^3$',fontsize=16,va='center',rotation='vertical')
plt.show()