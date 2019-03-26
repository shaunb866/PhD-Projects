import numpy as np
import h5py as h5
import eagle3 as E
import matplotlib.pyplot as plt




num_pp=np.load('Number_batch.npy')
n_p=np.load('Number_process.npy')
times=np.load('Times.npy')

plt.figure()
for i in range(len(n_p)):
	plt.plot(num_pp,times[i,:],label=str(n_p[i]))
	print('Minimum Time for '+str(n_p[i])+' is '+str(np.min(times[i,:]))+' using '+str(num_pp[np.argmin(times[i,:])]))

plt.xlabel('Number of particles in batch')
plt.ylabel('Time')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
