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
color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

rho_gauss=np.empty((nsims,len(mass_range)-1,low_len))
rho_hist=np.empty((nsims,len(mass_range)-1,low_len))
vel_gauss=np.empty((nsims,len(mass_range)-1,low_len,3))
vel_hist=np.empty((nsims,len(mass_range)-1,low_len,3))
num_halo=np.empty((nsims,len(mass_range)-1))
m_less=np.empty((nsims,len(mass_range)-1,low_len))
r_conv=np.empty((nsims,len(mass_range)-1))
r=np.load('r_0.0_12.0_12.5.npy')


for i in range(nsims):
	for j in range(len(label)-1):
		rho_gauss[i,j,:]=np.load('rho_gauss_%.1f_%.1f_%.1f.npy'%(cross_sections[i],label[j],label[j+1]))
		rho_hist[i,j,:]=np.load('rho_squar_%.1f_%.1f_%.1f.npy'%(cross_sections[i],label[j],label[j+1]))
		vel_gauss[i,j,:,:]=np.load('sig_gauss_%.1f_%.1f_%.1f.npy'%(cross_sections[i],label[j],label[j+1]))
		vel_hist[i,j,:,:]=np.load('sig_squar_%.1f_%.1f_%.1f.npy'%(cross_sections[i],label[j],label[j+1]))

		num_halo[i,j]=np.load('nhalo_%.1f_%.1f_%.1f.npy'%(cross_sections[i],label[j],label[j+1]))
		m_less[i,j,:]=np.load('m_%.1f_%.1f_%.1f.npy'%(cross_sections[i],label[j],label[j+1]))

		r_conv[i,j]=np.load('r_conv_%.1f_%.1f_%.1f.npy'%(cross_sections[i],label[j],label[j+1]))





#plot radial profiles
fig=plt.figure()
gs=gridspec.GridSpec(2,3)
gs.update(wspace=0.0, hspace=0.0)

ax=[]
for i in range(len(label)-1):
	ax.append(plt.subplot(gs[i]))

for i in range(nsims):
	for j in range(len(label)-1):
		#ax[j].plot(r,rho_hist[i,j,:]/num_halo[i,j],color=color[i],linestyle=':',lw=2.0)
		ax[j].plot(r,rho_gauss[i,j,:]/num_halo[i,j],color=color[i],linestyle='-',lw=1.5)
		
		ax[j].set_xscale('log')
		ax[j].set_yscale('log')
		ax[j].set_xlim(np.min(r),np.max(r))
		ylow=10**(-3)
		yhigh=10**(2.5)
		ax[j].set_ylim(ylow,yhigh)
		ax[j].text(0.05,0.05,'log M$_{\\rm{200c}}$='+str(label[i])+' -'+str(label[i+1]),transform=ax[i].transAxes,fontsize=16)
		ax[j].plot(np.array([1,1]),np.array([ylow,yhigh]),'--',color='gray',alpha=0.5)
		ax[j].plot(np.array([np.max(r_conv[:,j]),np.max(r_conv[:,j])]),np.array([ylow,yhigh]),'--',color='black')


ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[4].set_yticklabels([])

for i in range(nsims):
	ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color=color[i],label='$\\sigma=%.1f$'%(cross_sections[i]),lw=2.0)

#ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k-',label='Guassian Kernel',lw=2.0)
#ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k:',label='Histogram',lw=2.0)
ax[0].legend(loc='best')
ax[0].legend(frameon=False)

fig.text(0.5,0.04,'$R$',fontsize=20,ha='center')
fig.text(0.04,0.5,'$\\rho$',fontsize=20,va='center',rotation='vertical')



#plot velocity profiles
fig=plt.figure()
gs=gridspec.GridSpec(2,3)
gs.update(wspace=0.0, hspace=0.0)

ax=[]
for i in range(len(label)-1):
	ax.append(plt.subplot(gs[i]))

for i in range(nsims):
	for j in range(len(label)-1):
		#ax[j].plot(r,vel_hist[i,j,:,0],color=color[i],linestyle=':',lw=2.0)
		ax[j].plot(r,vel_gauss[i,j,:,0],color=color[i],linestyle='-',lw=1.5)
		
		ax[j].set_xscale('log')
		#ax[j].set_yscale('log')
		ax[j].set_xlim(np.min(r),np.max(r))
		ylow=0.5
		yhigh=0.9
		ax[j].set_ylim(ylow,yhigh)
		ax[j].text(0.05,0.05,'log M$_{\\rm{200c}}$='+str(label[i])+' -'+str(label[i+1]),transform=ax[i].transAxes,fontsize=16)
		ax[j].plot(np.array([1,1]),np.array([ylow,yhigh]),'--',color='gray',alpha=0.5)
		ax[j].plot(np.array([np.max(r_conv[:,j]),np.max(r_conv[:,j])]),np.array([ylow,yhigh]),'--',color='black')
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[4].set_yticklabels([])

for i in range(nsims):
	ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color=color[i],label='$\\sigma=%.1f$'%(cross_sections[i]),lw=2.0)

#ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k-',label='Guassian Kernel',lw=2.0)
#ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k:',label='Histogram',lw=2.0)
ax[0].legend(loc='best')
ax[0].legend(frameon=False)

fig.text(0.5,0.04,'$R$',fontsize=20,ha='center')
fig.text(0.04,0.5,'$\\sigma$',fontsize=20,va='center',rotation='vertical')



#plot entropy profiles
fig=plt.figure()
gs=gridspec.GridSpec(2,3)
gs.update(wspace=0.0, hspace=0.0)

ax=[]
for i in range(len(label)-1):
	ax.append(plt.subplot(gs[i]))

for i in range(nsims):
	for j in range(len(label)-1):
		#ax[j].plot(r,(rho_hist[i,j,:]/num_halo[i,j]/vel_hist[i,j,:,0]**3)**(-2/3),color=color[i],linestyle=':',lw=2.0)
		ax[j].plot(r,(rho_gauss[i,j,:]/num_halo[i,j]/vel_gauss[i,j,:,0]**3)**(-2/3),color=color[i],linestyle='-',lw=1.5)
		
		ax[j].set_xscale('log')
		ax[j].set_yscale('log')
		ax[j].set_xlim(np.min(r),np.max(r))
		ylow=10**(-2.5)
		yhigh=10**(2)
		ax[j].set_ylim(ylow,yhigh)
		ax[j].text(0.05,0.05,'log M$_{\\rm{200c}}$='+str(label[i])+' -'+str(label[i+1]),transform=ax[i].transAxes,fontsize=16)
		ax[j].plot(np.array([1,1]),np.array([ylow,yhigh]),'--',color='gray',alpha=0.5)
		ax[j].plot(np.array([np.max(r_conv[:,j]),np.max(r_conv[:,j])]),np.array([ylow,yhigh]),'--',color='black')


ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[4].set_yticklabels([])

for i in range(nsims):
	ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),color=color[i],label='$\\sigma=%.1f$'%(cross_sections[i]),lw=2.0)

#ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k-',label='Guassian Kernel',lw=2.0)
#ax[0].plot(np.ones(2)*(-1),np.ones(2)*(-1),'k:',label='Histogram',lw=2.0)
ax[0].legend(loc='best')
ax[0].legend(frameon=False)

fig.text(0.5,0.04,'$R$',fontsize=20,ha='center')
fig.text(0.04,0.5,'$S$',fontsize=20,va='center',rotation='vertical')




plt.show()