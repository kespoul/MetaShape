"""
Created on Tue Sep 17 09:05:22 2024

@author: kiesan

X = Metasurface width
Tx = Target plane width
N = # of pixels. Should be enough to have the width of each pixel = metasurface period
L = Distance between  metasurface and target
lbd0 = Wavelength of incident light
dx and dy = offset in pixels of light source center with respect to the metasurface. Simulate misalignment
z = distance from lightsource to metasurface. 
"""

#------------------Import packages --------------------------------------------

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy import *
import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

#Import GetPhase functions
from getPhase_BeamProp import *
from BasicFunctions import *

#------------------------------------------------------------------------------

#Define symbols for Sympy
x = Symbol('x')
tx = Symbol('tx')

#------------------------------------------------------------------------------

# Setup the problem:
X = 4       # [mm] metasurface width
Tx = X  # [mm] Target plane width
N =8000   # Number of pixels. Should be X/(metasurface period)
L = 200     # [mm] Distance between metasurface and target
z = 1       # Distance from light source to metasurface [mm]
lbd0 = 635  # [nm] lightsource wavelength
dx = 0      # Lightsource misalignment in pixels x-axis
dy = 0      # Lightsource misalignment in pixels y-axis

#------------------------------------------------------------------------------

#Extract problem dimensions for plotting
x_span, tx_span, x_init, tx_init, xx, yy, txx, tyy = Grid2D(X, Tx, N)

#------------------------------------------------------------------------------

# Input: Laser source as measured and fitted
theta = np.zeros(x_init.shape) # assume perfect collimation
#theta = np.linspace(-DivAng, DivAng, N) # Beam incidence angle
#I0 = 166.1
#I = I0*exp(-((x)/2.205)**2)
I0 = 166.1
I = I0*exp(-((x)/2.205)**2)

#Output: Double Gaussian (in 1D)
a = 0.8
b = Tx/3.5
c = 0.2
offset = 0.1 # Add small offset to avoid 0-valued pixels
E = a*(exp(-((tx-b)/c)**2)+exp(-((tx+b)/c)**2))+offset # Target intensity profile

#------------------------------------------------------------------------------

# Calculate phase profile
Fi, F, phi, phi_lbd0, k_xf_lbd0, I2d, E2d, tx_sol, x_sol, Start, Stop = GetPhase2D(X, Tx, L, N, I, E, lbd0, x, tx, theta)

# Discretize in phase levels and corresponding beam widths:
Plevels = [0.51, 1.19, 1.89, 2.62, 3.3, 4, 4.71, 5.41];
Wlevels = [148, 193, 220, 245, 265, 287, 382, 422];

phi_lbd0 = phi_lbd0%(2*np.pi)
phi_levels = np.zeros(phi_lbd0.shape)
width_levels = np.zeros(phi_lbd0.shape)

for i in range(phi_lbd0.shape[0]):
    for j in range(phi_lbd0.shape[0]):
        if np.isnan(phi_lbd0[i,j]):
            phi_levels[i,j] = 0
            width_levels[i,j] = 0;
        else:
            Test = abs(Plevels-phi_lbd0[i,j])
            ind = np.argmin(Test)
            phi_levels[i,j] = Plevels[ind]
            width_levels[i,j] = Wlevels[ind]


#Propagate field
Fi, F = BeamProp2D(phi_levels, F, Fi, I2d, L, Start, Stop, dx, dy)

# Extraxt simulated intensity profile at target plane
Int = F.get_intensity()

#------------------------------------------------------------------------------

Lim1 = int((Int.shape[0]-I2d.shape[0])/2-1)
Lim2 = int((Int.shape[0]-I2d.shape[0])/2-1+N)

# Plot Intensity profiles
fig, (ax1, ax2) = plt.subplots(ncols = 2, layout='constrained')

im = ax1.imshow(I2d, cmap = 'inferno', vmin=0, vmax=550)
ax1.set_title("Incident Field")
cb = fig.colorbar(im, ax = ax1)

im = ax2.imshow(Int[Lim1:Lim2,Lim1:Lim2], cmap = 'inferno', vmin=0, vmax=550)
ax2.set_title("Propagated field")

cb = fig.colorbar(im, ax = ax2)

plt.show()

#------------------------------------------------------------------------------
center = int(N/2-1)
center_diffractsim = int((Int.shape[0]/2-1))

# Plot incident field, phase profile and field at target plane
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = [5,2], layout='constrained')

ax[0].plot(x_init,I2d[:,center])
ax[0].set_xlabel('x [mm]')
ax[0].set_ylabel('Intensity a.u.')
ax[0].title.set_text('Input')

ax[1].plot(x_sol,phi_levels[:,center])
ax[1].set_xlabel('x [mm]')
ax[1].set_ylabel('Phase [rad]')
ax[1].title.set_text('Phase')

ax[2].plot(F.xx[0,:]*1e3,Int[:,center_diffractsim],label = 'Sim')
ax[2].plot(tx_init,E2d[:,center],label = 'Design')
ax[2].set_xlim([-1.1*Tx/2, 1.1*Tx/2])
ax[2].set_ylim([0, 550])
ax[2].set_xlabel('tx [mm]')
ax[2].set_ylabel('Intensity a.u.')
ax[2].legend()
ax[2].title.set_text('Output')

# Plot Continous and leveled phase:
fig2, ax2 = plt.subplots(nrows = 1, ncols = 2, figsize = [5,2], layout='constrained')

ax2[0].plot(x_sol,phi_lbd0[:,center])
#ax2[0].plot(x_sol,phi_lbd0[:,center]*1.5)
ax2[0].set_xlabel('x [mm]')
ax2[0].set_ylabel('Phase [rad]')
ax2[0].set_xlim([0, 1.1*X/2])
ax2[0].title.set_text('Continous')

ax2[1].plot(x_sol,phi_levels[:,center])
ax2[1].set_xlabel('x [mm]')
ax2[1].set_ylabel('Phase [rad]')
ax2[1].set_xlim([0, 1.1*X/2])
ax2[1].title.set_text('8 levels')

fig2.suptitle('Half phase profile')

#------------------------------------------------------------------------------
# Plot 2D phase profile - continous and leveled
fig, (ax3,ax4) = plt.subplots(ncols = 2, layout='constrained')

im = ax3.imshow(phi_lbd0, cmap = 'inferno', vmin=0, vmax=6.5)
ax1.set_title("Incident Field")
cb = fig.colorbar(im, ax = ax3)

im = ax4.imshow(phi_levels, cmap = 'inferno', vmin=0, vmax=6.5)
ax1.set_title("Incident Field")
cb = fig.colorbar(im, ax = ax4)

#------------------------------------------------------------------------------
# Plot longitudinal profile - WARNING: VERY SLOW. Decrease "steps" to increase speed and decrease accuracy.
longitudinal_profile_rgb, longitudinal_profile_E, extent = Fi.get_longitudinal_profile( start_distance = 0*mm , end_distance = 350*mm , steps = 50) 
long = Fi.plot_longitudinal_profile_intensity(longitudinal_profile_E = longitudinal_profile_E, extent = extent)














