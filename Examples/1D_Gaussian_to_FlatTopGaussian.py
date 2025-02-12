# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:17:03 2024

@author: kiesan

Description: This example calculates the necessary phase profile to map an input beam with a Gaussian intensity profile 
to a flattop Gaussian intensity profile at the target plane (All in 1D). It then propagates the beam with the calculated
phase profile, in order to simulate the resulting intensity at the target plane.

X = Metasurface width
Tx = Target plane width
N = # of pixels. Should be enough to have the width of each pixel = metasurface period
L = Distance between  metasurface and target
lbd0 = Wavelength of incident light
dx and dy = offset in pixels of light source center with respect to the metasurface. Simulate misalignment
z = distance from light source to metasurface. Used to calculate beamwidth.
w0 = beam waist
zr = Rayleigh range for the gaussian beam
DivAng = divergenve angle
theta = Resulting incidence angles at the metasurface
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
X = 5       # [mm] metasurface width
Tx = 2*X  # [mm] Target plane width
N = 10000   # Number of pixels. Should be X/(metasurface period)
L = 100     # [mm] Distance between metasurface and target
z = 5 #Distance from light source to metasurface [mm]
lbd0 = 600  # [nm] lightsource wavelength
lbd = lbd0*10**(-6)
dx = 0      # Lightsource misalignment in pixels x-axis
dy = 0      # Lightsource misalignment in pixels y-axis

#------------------------------------------------------------------------------

#Extract problem dimensions for plotting
x_span, tx_span, x_init, tx_init = Grid(X, Tx, N)

#------------------------------------------------------------------------------

# Input: Gaussian intensity profile
w0 = 2 # Beam waist [mm]
n = 1 # Refractive index of propagation medium
zr = np.pi*(w0**2)*n/lbd #Rayleigh range
w = w0*sqrt(1+(z/zr)**2)
DivAng = lbd/(np.pi*n*w0) #Divergence angle
theta = np.linspace(-DivAng, DivAng, N) # Beam incidence angle
I0 = 1
I = I0*((w0/w)**2)*exp(-(2*x**2)/(w**2))


# Output: Flat top Gaussian profile
nn = 10 #Order of supergaussian
E =I0*exp(-2*(tx/w0)**nn)+0.001 # Target intensity add a small offset to avoid 0-valued pixels

#------------------------------------------------------------------------------

# Calculate phase profile
Fi, F, phi, phi_lbd0, k_xf_lbd0, I1d, E1d, tx_sol, x_sol, Start, Stop = GetPhase(X, Tx, L, N, I, E, lbd0, x, tx, theta)

# Propagate field
Fi, F = BeamProp(k_xf_lbd0, F, Fi, I1d, L, Start, Stop, dx)

#------------------------------------------------------------------------------

# Plot longitudinal profile - 2*L propagation
longitudinal_profile_rgb, longitudinal_profile_E, extent = Fi.get_longitudinal_profile( start_distance = 0*mm , end_distance = 2*L*mm , steps = 200) 
long = Fi.plot_longitudinal_profile_intensity(longitudinal_profile_E = longitudinal_profile_E, extent = extent)

#------------------------------------------------------------------------------

# Extraxt simulated intensity profile at target plane
Int = F.get_intensity()

# Plot incident field, phase profile and field at target plane
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = [5,2], layout='constrained')

ax[0].plot(x_init,I1d)
ax[0].set_xlabel('x [mm]')
ax[0].set_ylabel('Intensity a.u.')
ax[0].title.set_text('Input')

ax[1].plot(x_sol,phi_lbd0)
ax[1].set_xlabel('x [mm]')
ax[1].set_ylabel('Phase [rad]')
ax[1].title.set_text('Phase')

ax[2].plot(F.xx[0,:]*1e3,Int[0],label = 'Sim')
ax[2].plot(tx_init,E1d,label = 'Design')
ax[2].set_xlim([-1.1*Tx/2, 1.1*Tx/2])
ax[2].set_xlabel('tx [mm]')
ax[2].set_ylabel('Intensity a.u.')
ax[2].legend()
ax[2].title.set_text('Output')

# Plot unwrapped and wrapped phase:
fig2, ax2 = plt.subplots(nrows = 1, ncols = 2, figsize = [5,2], layout='constrained')

ax2[0].plot(x_sol,phi_lbd0)
ax2[0].set_xlabel('x [mm]')
ax2[0].set_ylabel('Phase [rad]')
ax2[0].set_xlim([0, 1.1*X/2])
ax2[0].title.set_text('Unwrapped')

ax2[1].plot(x_sol,phi_lbd0%(2*np.pi))
ax2[1].set_xlabel('x [mm]')
ax2[1].set_ylabel('Phase [rad]')
ax2[1].set_xlim([0, 1.1*X/2])
ax2[1].title.set_text('Wrapped')

fig2.suptitle('Half phase profile')
























