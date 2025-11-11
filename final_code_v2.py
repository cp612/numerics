# -*- coding: utf-8 -*-
"""
Run this file to produce all the plots for the numerics report
"""

import numpy as np
import matplotlib.pyplot as plt

from numerics import advect as advect
from numerics import convergence_test as convergence_test

#%% Conservation of Mass and variance

# number of SL timesteps 
SL_timesteps=75   

# advect initial conditions
FTBS_data = advect(1.0, 100,150,  1.0,scheme='FTBS',use_square=True)
CTCS_data = advect(1.0, 100,150,  1.0,scheme='CTCS',use_square=True)
SL_data= advect(1.0, 100,SL_timesteps,  1.0,scheme='SL',use_square=True)
CN_data= advect(1.0, 100,150,  1.0,scheme='CN',use_square=True)


# Plot conservation of mass and variance for each of the four schemes
fig,axes=plt.subplots(nrows=2,ncols=4,figsize=(12,6),sharex=True,sharey=False)   
time=np.arange(0,1,1/150)
time_SL=np.arange(0,1,1/SL_timesteps)
axes[0,0].plot(time,FTBS_data[2],label='FTBS',color='r')
axes[1,0].plot(time,FTBS_data[3],label='FTBS',color='r')
axes[0,1].plot(time,CTCS_data[2],label='CTCS',color='b')
axes[1,1].plot(time,CTCS_data[3],label='CTCS',color='b')
axes[0,2].plot(time_SL,SL_data[2],label='SL',color='orange')
axes[1,2].plot(time_SL,SL_data[3],label='SL',color='orange')
axes[0,3].plot(time,CN_data[2],label='CN',color='green')
axes[1,3].plot(time,CN_data[3],label='CN',color='green')

# Styling
for ax in range(4):
    axes[0,ax].legend()
    axes[1,ax].legend()
    axes[0,ax].set_ylim(0.38,0.42)
    axes[1,ax].set_ylim(0.0,0.25)
    axes[1,ax].set_xlabel('Time (s)',fontsize=16)

for ax in range(3):
    axes[0,ax+1].set_yticks([])
    axes[1,ax+1].set_yticks([])
    
    
axes[0,0].set_ylabel('Mass', fontsize=16)   
axes[1,0].set_ylabel('Variance', fontsize=16)
axes[0,0].set_ylim(0.38,0.42)
axes[1,3].set_xlabel('Time (s)',fontsize=16)
axes[1,0].set_ylim(0.,0.25)

plt.tight_layout()
plt.show()

#%% Filter plots
import matplotlib as mpl

# run the three filter cases
no_filter = advect(1, 100,125, 1.0, scheme='CTCS',use_square=True,alpha=0.)
half_filter= advect(1, 100,125, 1.0,scheme='CTCS',use_square=True,alpha=0.2)
full_filter= advect(1, 100,125, 1.0,scheme='CTCS',use_square=True,alpha=0.3)


# plot Hovmoller diagrams
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(10,4),sharey=True,layout='constrained')
im1=axes[0].imshow(no_filter[1].T,vmin=-0.2,vmax=1.2)
im2=axes[1].imshow(half_filter[1].T,vmin=-0.2,vmax=1.2)
im3=axes[2].imshow(full_filter[1].T,vmin=-0.2,vmax=1.2)

# colourbar
cbar=fig.colorbar(im3)
cbar.set_label(r'$\phi$',fontsize=18)
cbar.ax.tick_params(labelsize=12)

# styling
for i in range (3):
    axes[i].tick_params('both',labelsize=12)
    axes[i].set_aspect('auto')
    axes[i].set_xlabel('x',fontsize=18)


axes[0].invert_yaxis()
axes[0].set_ylabel('Timestep',fontsize=18)

plt.show()
#%% Stability 

# run stability test cases for all four schemes
stable_FTBS = advect(2, 50,200, 1.0,scheme='FTBS',use_square=True,alpha=0.0)
unstable_FTBS = advect(2, 110,200, 1.0,scheme='FTBS',use_square=True,alpha=0.0)
stable_CTCS=advect(2, 50,200, 1.0,scheme='CTCS',use_square=True,alpha=0.0)
unstable_CTCS=advect(2, 110,200, 1.0,scheme='CTCS',use_square=True,alpha=0.0)
stable_SL = advect(1, 300,55, 1.0,scheme='SL',use_square=True,alpha=0.)
stable_CN = advect(1, 300,55, 1.0,scheme='CN',use_square=True,alpha=0.2)

# plotting
fig, axes = plt.subplots(nrows=3, ncols=2,figsize=(10,6),sharex=True,sharey=True,layout='constrained')

im1=axes[0,0].imshow(stable_FTBS[1].T,vmin=-0.2,vmax=1.2,extent=[0,1,2,0])
im2=axes[0,1].imshow(unstable_FTBS[1].T,vmin=-0.2,vmax=1.2,extent=[0,1,2,0])
im3=axes[1,0].imshow(stable_CTCS[1].T,vmin=-0.2,vmax=1.2,extent=[0,1,2,0])
im4=axes[1,1].imshow(unstable_CTCS[1].T,vmin=-0.2,vmax=1.2,extent=[0,1,2,0])
im5=axes[2,0].imshow(stable_SL[1].T,vmin=-0.2,vmax=1.2,extent=[0,1,1,0])
im6=axes[2,1].imshow(stable_CN[1].T,vmin=-0.2,vmax=1.2,extent=[0,1,1,0])


# styling
for i in range (3):
    axes[i,0].set_ylabel('Time (s)',fontsize=18)
    for j in range  (2):
        axes[i,j].set_aspect('auto')
for j in range (2):
    for i in range(2):
        axes[i,j].set_xlabel('',fontsize=18)
        axes[2,1].set_xlabel('x',fontsize=18)
        axes[2,0].set_xlabel('x',fontsize=18)
        axes[i,j].tick_params('both',labelsize=12)
        
axes[0,0].invert_yaxis()

# colourbar
cbar=fig.colorbar(im2)
cbar.set_label(r'$\phi$',fontsize=18)
cbar=fig.colorbar(im4)
cbar.set_label(r'$\phi$',fontsize=18)
cbar=fig.colorbar(im6)
cbar.set_label(r'$\phi$',fontsize=18)
cbar.ax.tick_params(labelsize=12)

plt.show()

#%%
# compute errors for convergence experiment
CTCS_errors=convergence_test('CTCS',alpha_value=0.00)
FTBS_errors=convergence_test('FTBS',alpha_value=0.00)
SL_errors=convergence_test('SL')
CN_errors=convergence_test('CN')

# spatial steps for plotting
nx_list = np.array([800, 1600, 3200, 6400])
dx_list = 1.0 / nx_list
cn_dx=1/np.array([100, 200, 400, 800])

# plotting
plt.figure(figsize=(6, 5))
plt.loglog(dx_list,CTCS_errors[0], 'o-', label='CTCS')
plt.loglog(dx_list,FTBS_errors[0], 'o-', label='FTBS')
plt.loglog(dx_list,SL_errors[0], 'o-', label='SL')
plt.loglog(cn_dx,CN_errors[0], 'o-', label='CN')
plt.xlabel('Grid spacing Δx')
plt.ylabel('L₂ error')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend()
plt.show()


#%% Order of accuracy against Asselin coefficient
alpha=np.arange(0.01,0.21,0.01)
filter_errors=[]

# compute order of accuracy for different filter strengths
for a in alpha:
    CTCS_errors,CTCS_order=convergence_test('CTCS',alpha_value=a)
    filter_errors.append(CTCS_order)
    print(a)

# plotting
plt.plot(alpha,filter_errors)
plt.xlabel(r"$\gamma$")
plt.ylabel("Order of Accuracy")
plt.show()

