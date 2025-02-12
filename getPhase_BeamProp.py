# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:24:35 2024

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
        k_xf : Resulting phase profile after metasurface (to use with Diffractsim). Wavelength independent
        k_xf_lbd0 : Resulting phase profile after metasurface (to use with Diffractsim). Wavelength dependent
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
        GetPhase : Calculate metasurface phase profile (1D)
        GetPhase2D : Calculate metasurface phase profile (2D)
        BeamProp : Propagate field to get resulting target intensity (1D)
        BeamProp2D : Propagate field to get resulting target intensity (2D)

"""
#Import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumtrapz
from sympy import *
from BasicFunctions import *
import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration


def GetPhase(X, Tx, L, N, I, E, lbd0, x, tx, theta):
    #---------Setup Grid-------------------------------------------------------

    x_span, tx_span, x_init, tx_init = Grid(X, Tx, N)
       
    #---------Setup intensity ditributions-------------------------------------
    
    #Normalize target distribution
    Enorm = NormInt(I, E, X, Tx, x, tx)
    
    #Create lambda functions
    dtxdx = lambdify([x, tx],I/Enorm)
    Inum = lambdify(x,I)
    Enum = lambdify(tx,Enorm)
    
    #---------Initialize fields for Diffractsim--------------------------------

    Fi, F, Start, Stop = DiffractsimInit1D(X, Tx, N, L, lbd0)

    #---------Solve Equation---------------------------------------------------
    
    #Set initial condition to the left boundary condition:
    y0 = np.empty(1)
    y0[0] = tx_span[0]
    
    #Solve equation using Runge-Kutta of order 5
    sol = solve_ivp(dtxdx, x_span, y0, t_eval = x_init, method = 'RK45', rtol=1e-10, atol = 1e-8)
    
    #Extract solution
    x_sol = sol.t
    tx_sol = sol.y[0,:]
    
    #---------Define intensity arrays------------------------------------------
    
    #Intensity arrays
    I1d = np.zeros(x_init.shape)
    I1d[:] = Inum(x_init)
    E1d = np.zeros(tx_init.shape)
    E1d[:] = Enum(tx_init)
    
    #---------Calculate phase profiles-----------------------------------------
    
    # Calculate phase and phase gradient
    phix = ((tx_sol - x_sol) / np.sqrt(L**2 + (tx_sol-x_sol)**2))-np.sin(theta)
    phi = cumtrapz(phix,x_sol, initial = 0)
    k_xf = cumtrapz(phix + np.sin(theta),x_sol, initial = 0)
    
    k = 2*np.pi/(lbd0*1e-6)
    phi_lbd0 = k*phi
    k_xf_lbd0 = k*k_xf
    
    return Fi, F, phi, phi_lbd0, k_xf_lbd0, I1d, E1d, tx_sol, x_sol, Start, Stop

def GetPhase2D(X, Tx, L, N, I, E, lbd0, x, tx, theta):
    #---------Setup Grid-------------------------------------------------------

    x_span, tx_span, x_init, tx_init, xx, yy, txx, tyy = Grid2D(X, Tx, N)
    
    #---------Setup intensity ditributions-------------------------------------   
    
    #Normalize target distribution
    Enorm = NormInt2D(I, E, X, Tx, x, tx)
    
    #Create lambda functions
    dtxdx = lambdify([x, tx],abs(x)*I/(abs(tx)*Enorm))
    Inum = lambdify(x,I)
    Enum = lambdify(tx,Enorm)
     
    #---------Initialize fields for Diffractsim--------------------------------

    Fi, F, Start, Stop = DiffractsimInit2D(X, Tx, N, L, lbd0)
    
    #---------Solve Equation---------------------------------------------------
  
    #Set initial condition to the left boundary condition:
    y0 = np.empty(1)
    y0[0] = tx_span[0]
    
    #Solve equation using Runge-Kutta of order 5
    sol = solve_ivp(dtxdx, x_span, y0, t_eval = x_init, method = 'RK45', rtol=1e-10, atol = 1e-8)
    
    #Extract solution
    x_sol = sol.t
    tx_sol = sol.y[0,:]
    
    #---------Calculate phase profile------------------------------------------
    
    # Calculate phase and phase gradient
    phix = ((tx_sol - x_sol) / np.sqrt(L**2 + (tx_sol-x_sol)**2))-np.sin(theta)
    phi = cumtrapz(phix,x_sol, initial = 0)
    k_xf = cumtrapz(phix + np.sin(theta),x_sol, initial = 0)
    
    # Interpolate to get 2D phase profile and intensity profiles
    rad, mask, phi_2d = RadialInterp(xx,yy,X,x_init,phi)
    rad_f, mask_f, phi_2d_f = RadialInterp(txx,tyy,Tx,tx_init,phi)
    rad_k, mask_k, k_xf_2d = RadialInterp(xx,yy,X,x_init,k_xf)
    
    k = 2*np.pi/(lbd0*1e-6)
    phi_lbd0 = k*phi_2d
    phi_lbd0[mask] = 0
        
    #Interpolate to get resulting phase profile (k_xf)
    k_xf_lbd0 = k*k_xf_2d
    k_xf_lbd0[mask] = 0
    
    I2d = np.zeros(phi_2d.shape)
    I2d[:] = Inum(rad)
    I2d[mask] = 0
    
    E2d = np.zeros(phi_2d_f.shape)
    E2d[:] = Enum(rad_f)
    E2d[mask_f] = 0
    
    return Fi, F, phi, phi_lbd0, k_xf_lbd0, I2d, E2d, tx_sol, x_sol, Start, Stop

def BeamProp(k_xf_lbd0, F, Fi, I1d, L, Start, Stop, dx):
    
    #Define incident field   
    F.E = np.zeros(F.xx.shape,dtype='complex')
    F.E[0,Start+dx:Stop+dx] = np.sqrt(I1d)
    F.E[0,Start:Stop] *= np.exp(1j*(k_xf_lbd0))
    F.E[0,0:(Start-1)] = 0
    F.E[0,(Stop+1):-1] = 0
        
    #Save field for longitudinal profile
    Fi.E = F.E
    
    # Propagate field
    F.propagate(L*mm, scale_factor = 1)
    
    return Fi, F

def BeamProp2D(k_xf_lbd0, F, Fi, I2d, L, Start, Stop, dx, dy):

    #Define incident field   
    F.E = np.zeros(F.xx.shape,dtype='complex')
    F.E[Start+dx:Stop+dx,Start+dy:Stop+dy] = np.sqrt(I2d)
    F.E[Start:Stop,Start:Stop] *= np.exp(1j*k_xf_lbd0)

    
    #Save field for longitudinal profile
    Fi.E = F.E
    
    # Propagate field
    F.propagate(L*mm, scale_factor = 1)
    
    return Fi, F
    
    
    
    
    
    
    
    
    
