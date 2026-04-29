import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("dipole_central_cor_mean_v3.txt")

A = len(data[:,0])

data = data[:A//5*4,:]

data[:,0] *= 1e-6
data[:,1] -= data[:,1].min()
data[:,2] -= data[:,2].min()
data[:,3] -= data[:,3].min()
data[:,1] /= data[0,1]
data[:,2] /= data[0,2]
data[:,3] /= data[0,3]



fig, ax = plt.subplots(figsize =(7.7, 4.8))


ax.plot(data[:,0], data[:,1], label = 'x')
ax.plot(data[:,0], data[:,2], label = 'y')
ax.plot(data[:,0], data[:,3], label = 'z')
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Autocorr (-)")
ax.set_title("LD central dipole autocorr")
ax.legend()
ax.grid()
ax.set_xscale('log')


plt.tight_layout()
plt.savefig("LD_Autocorr.png", dpi=300)
plt.show()
