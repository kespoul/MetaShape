# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 08:36:35 2024

@author: kiesan

Definitions
----------
Initial parameters:
    -----
    X : Metasurface width
    Tx : Width of target plane
    N : Number of discrete points
    L : Distance between metasurface and target plane
    I : input intensity function
    E : Output intensity function
    lbd0 : Operating wavelength
    dx : displacement of beam aling x-axis (in #pixels)
    dy : displacement of beam aling y-axis (in #pixels)
    x : Metasurface coordinate (defined as a Sympy symbol)
    tx : Target plane coordinate (defined as a Sympy symbol)

Calculated parameters
    -----
    Fi : Field at metasurface for longitundinal profile plotting
    F : Field at metasurface for propagation OR propagated field
    phi : Wavelength independent phase profile
    phi_lbd0 : Wavelength dependent phase profile
    k_xf : Wavelength independent phase profile for Diffractsim
    k_xf_lbd0 : Wavelength dependent phase profile for Diffractsim
    I1d : incident intensity array (1D)
    E1d : target intensity array (1D)
    I2d : incident intensity array (2D)
    E2d : target intensity array (2D)
    tx_sol : solution to diff eq (target coordinates)
    x_sol : x-coordinates corresponding to solution
    Start : Starting point for incident beam
    Stop : Stop point for incident beam

Functions
    -----
    Grid : setup metasurface and target plane grid (1d)
    Grid2D : setup metasurface and target plane grid (2d)
    DiffractsimInit1D : setup grid and E-field structures for Diffractsim (1d)
    DiffractsimInit2D : setup grid and E-field structures for Diffractsim (2d)
    NormInt : Normalize Target intensity distribution to incident intensity (1D)
    NormInt2D : Normalize Target intensity distribution to incident intensity (2D)
    RadialInterp : Perform a radial interpolatino of the phase profile to get 2D profile
    PhaseLevels : Disretize calculated phase profile into X phase levels (1D)
    PhaseLevels2D : Disretize calculated phase profile into X phase levels (2D)

"""
#Import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import interp1d
from sympy import *
import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration
from diffractsim import MonochromaticField, nm, mm, cm

# Functions for defining a grid for the problem (1D or 2D)
def Grid(X, Tx, N):
    x_span = np.array([-X/2 , X/2])
    tx_span = np.array([-Tx/2 , Tx/2])
    
    x_init = np.linspace(-X/2, X/2, num = N)
    tx_init = np.linspace(-Tx/2, Tx/2, num = N)
    return x_span, tx_span, x_init, tx_init

def Grid2D(X, Tx, N):
    x_span = np.array([-X/2 , X/2])
    tx_span = np.array([-Tx/2 , Tx/2])
    
    x_init = np.linspace(-X/2, X/2, num = N)
    y_init = x_init
    tx_init = np.linspace(-Tx/2, Tx/2, num = N)
    ty_init = tx_init
    
    xx, yy = np.meshgrid(x_init,y_init)
    txx, tyy = np.meshgrid(tx_init,ty_init)
    return x_span, tx_span, x_init, tx_init, xx, yy, txx, tyy

# Functions for setting up the field structures for diffractsim (1D or 2D)
def DiffractsimInit1D(X, Tx, N, L, lbd0):
    padding = 1 # Make sure to always have at least 1 mm of free space in the grid at both ends
    if X >= Tx:
       scale = (X+padding)/X
    elif Tx > X:
       scale = (Tx+padding)/X

    extent_x = X*scale
    extent_y = X*scale
    Nx = int(N*scale)
    Start = int((Nx-N)/2-1)
    Stop = Start+N
        
    #Initalize field structures
    Fi = MonochromaticField(wavelength = lbd0 * nm, extent_x=extent_x * mm, extent_y=extent_y * mm, Nx=Nx, Ny=1, intensity =0.01)
    F = MonochromaticField(wavelength = lbd0 * nm, extent_x=extent_x * mm, extent_y=extent_y * mm, Nx=Nx, Ny=1, intensity =0.01)
    return Fi, F, Start, Stop

def DiffractsimInit2D(X, Tx, N, L, lbd0):
    padding = 1 # Make sure to always have at least 1 mm of free space in the grid at both ends
    if X >= Tx:
       scale = (X+padding)/X
    elif Tx > X:
       scale = (Tx+padding)/X

    extent_x = X*scale
    extent_y = X*scale
    Nx = int(N*scale)
    Ny = Nx
    Start = int((Nx-N)/2-1)
    Stop = Start+N
        
    #Initalize field structures
    Fi = MonochromaticField(wavelength = lbd0 * nm, extent_x=extent_x * mm, extent_y=extent_y * mm, Nx=Nx, Ny=Ny, intensity =0.01)
    F = MonochromaticField(wavelength = lbd0 * nm, extent_x=extent_x * mm, extent_y=extent_y * mm, Nx=Nx, Ny=Ny, intensity =0.01)
    return Fi, F, Start, Stop

# Functions for normalizing the Target intensity to the incident intensity (Optimal Transport, 1D or 2D)
def NormInt(I, E, X, Tx, x, tx):
    I_int = Integral(I,(x,-X/2,X/2)).evalf()
    E_int = Integral(E,(tx,-Tx/2,Tx/2)).evalf()
    Enorm = (I_int/E_int)*E
    return Enorm

def NormInt2D(I, E, X, Tx, x, tx):
    I_int = Integral(abs(x)*I,(x,-X/2,X/2)).evalf()
    E_int = Integral(abs(tx)*E,(tx,-Tx/2,Tx/2)).evalf()
    Enorm = (I_int/E_int)*E
    return Enorm

# Function that performs a radial interpolation of a 1D function to create a 2D function (array form)
def RadialInterp(xx,yy,X,x_init,phi):
    rad = np.sqrt(xx**2 + yy**2)
    mask = rad > X/2
    rad[mask] = np.nan
    phi_interp = interp1d(x_init,phi)
    phi_2d = phi_interp(rad)
    return rad, mask, phi_2d

# Functions for creating discrete and evenly spaced phase levels instead of a continous profile (1D or 2D)
def PhaseLevels(phi_lbd0, num):
    Levels = np.linspace(0, 2, num = num, endpoint=True)
    Levels = Levels*np.pi

    phi_levels = np.zeros(phi_lbd0.shape)
    for i in range(phi_lbd0.shape[0]):
            if np.isnan(phi_lbd0[i]):
                phi_levels[i] = 0
            else:
                Test = abs(Levels-phi_lbd0[i])
                ind = np.argmin(Test)
                phi_levels[i] = Levels[ind]
    return phi_levels

def PhaseLevels2D(phi_lbd0, num):
    Levels = np.linspace(0, 2, num = num, endpoint=False)
    Levels = Levels*np.pi

    phi_levels = np.zeros(phi_lbd0.shape)
    for i in range(phi_lbd0.shape[0]):
        for j in range(phi_lbd0.shape[0]):
            if np.isnan(phi_lbd0[i,j]):
                phi_levels[i,j] = 0
            else:
                Test = abs(Levels-phi_lbd0[i,j])
                ind = np.argmin(Test)
                phi_levels[i,j] = Levels[ind]
    return phi_levels
    