# -*- coding: utf-8 -*-
"""
Functions for plots produced in the numerics report
"""

import numpy as np
import matplotlib.pyplot as plt


def initial_condition(x,use_square=True):
    '''
    Sets the initial condition for the advection problem: smooth bell
    with optional square wave

    Parameters
    ----------
    x (float): the spatial grid.
    use_square (bool): add a square wave to the initial condition
        

    Returns
    -------
    phi0 (array of floats): y values for the initial condition

    '''
    bell = np.where(x % 1.0 < 0.5, np.sin(2 * np.pi * x) ** 4, 0.0)
    square = np.where((x >= 0.7) & (x <= 0.9), 1.0, 0.0)
    if use_square:
        phi0= bell + square
    else:
        phi0= bell
    return phi0


def FTBS(phiOld,C):
    '''
    Performs forward-time backwards-space advection

    Parameters
    ----------
    phiOld (array of floats): phi at time n
    C (float): Courant number

    Returns
    -------
    phi (array of floats): phi at time n+1
    '''
    #np.roll enforces periodic boundary conditions automatically
    phi = phiOld - C * (phiOld - np.roll(phiOld, 1))
    return phi

def CTCS(phi_nm1, phi_n, C):
    '''
    Performs centred-time centred-space advection

    Parameters
    ----------
    phi_nm1 (array of floats): phi at time n-1
    phi_n (array of floats): phi at time n
    C (float): Courant number

    Returns
    -------
    (array of floats) returns phi at time n+1
    '''
    return phi_nm1 - C*(np.roll(phi_n, -1) - np.roll(phi_n, 1))


def semi_lagrangian_textbook(phi, C):
    '''
    Performs semi-lagrangian advection

    Parameters
    ----------
    phi (array of floats): phi at time n
    C (float): Courant number

    Returns
    -------
    phi_new (array of floats): phi at time n+1

    '''
    phi_new = np.zeros_like(phi)
    
    p = int(np.floor(C))          # integer part of Courant
    alpha = C - p                 # fractional part of Courant

    phi_new[:] = alpha * np.roll(phi, p+1) + (1 - alpha) * np.roll(phi, p) # Semi-Lagrangian interpolation
    
    return phi_new


def Crank_Nicholson(nx,C):
    '''
    Performs Crank-Nicholson advection

    Parameters
    ----------
    nx (int): number of timesteps
    C (flaot): Courant number

    Returns
    -------
    A (array of flaots): LHS matrix for Crank-Nicholson advection
    B (array of floats): RHS matrix for Crank-Nicholson advection

    '''
    # set up empty matrices
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nx))
    
    #fill matrices for Crank-Nicholson advection
    for i in range (nx):
        A[i,i] = 1.
        A[i, (i+1)%nx] = C/4
        A[i, (i-1)%nx] = -C/4

        B[i, i] = 1.
        B[i, (i+1)%nx] = -C/4
        B[i, (i-1)%nx] = C/4
    return A, B


def FTCS(phi,C):
    '''
    Performs forward-time centred-space

    Parameters
    ----------
    phi (array of floats): phi at time n-1
    C (float): courant number

    Returns
    -------
    (array of floats): phi at time n

    '''
    return phi - (C/2)*(np.roll(phi,-1)-np.roll(phi,1))
    

def advect(final_time, nx,nt, u, scheme='FTBS', alpha=0.05,use_square=True):
    '''
    Handles advection for all four schemes

    Parameters
    ----------
    final_time (float): how many seconds to advect
    nx (integer): number of grid points
    nt (integer): number of timesteps
    u (float): wind speed
    scheme (string): Name of the scheme The default is 'FTBS'.
    alpha (float): Asselin coefficientThe default is 0.05.
    use_square (bool): adds square wave to initial condition
    Returns
    -------
    l2_error (float): L2 error norm for the scheme
    phi_hov (array of floats): stores values of phi at time n and gridpoint x
    mass (array of floats): mass at each timestep
    variance (array of floats): variance at each timestep

    '''
    # time and space steps 
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    dx = 1.0 / nx
    dt = final_time/nt
    # Courant number
    C=u*(dt/dx)
    
    # phi for analytical
    phi0 = initial_condition(x,use_square)
    # phi for numerical
    phiOld = phi0.copy()
    
    # store values
    phi_hov=np.zeros((nx,nt))
    variance = []
    mass=[]
    
    if scheme == 'FTBS':
        for n in range(nt):
            # advect
            phi = FTBS(phiOld,C)
            # move forwards delta t in time
            phiOld[:] = phi
            
            phi_hov[:,n]=phi
            mass.append(np.sum(phi)*dx)
            variance_phi = np.mean((phiOld - np.mean(phiOld))**2)
            variance.append(variance_phi)

                
    elif scheme == 'FTCS':
        for n in range(nt):
            # advect
            phi= FTCS(phiOld,C)
            # move forwards delta t in time
            phiOld[:]=phi
            
            phi_hov[:,n]=phi
            mass.append(np.sum(phi)*dx)
            variance_phi = np.mean((phiOld - np.mean(phiOld))**2)
            variance.append(variance_phi)

    elif scheme == 'CTCS':
        # first timestep with FTCS
        phi_nm1 = phiOld.copy()        
        phi_n = FTCS(phiOld,C)  
        
        mass.append(np.sum(phi_n)*dx)
        variance_phi = np.mean((phi_n - np.mean(phi_n))**2)
        variance.append(variance_phi)
        
        
        for n in range(1, nt):
            # advect with CTCS
            phi_np1 = CTCS(phi_nm1, phi_n, C)

            # apply Robert-Asselin filter
            if alpha > 0.0:
                phi_n += alpha * (phi_np1 - 2 * phi_n + phi_nm1)
           

            # move forwards delta t in time
            phi_nm1[:] = phi_n
            phi_n[:] = phi_np1
            
            phi_hov[:,n]=phi_n
            mass.append(np.sum(phi_np1)*dx)
            variance_phi = np.mean((phi_n - np.mean(phi_n))**2)
            variance.append(variance_phi)

            
        phi = phi_n
        
    elif scheme == 'SL':
        for n in range (nt):
            # advect
            phi_np1 = semi_lagrangian_textbook(phiOld, C)
            # move forwards delta t in time
            phiOld[:] = phi_np1
            
            phi_hov[:,n]=phi_np1
            variance_phi = np.mean((phi_np1 - np.mean(phi_np1))**2)
            variance.append(variance_phi)
            mass.append(np.sum(phi_np1)*dx)

        
        
        phi=phiOld
    elif scheme == 'CN':
        # return matrices
        A, B = Crank_Nicholson(nx, C)
        
        for n in range (nt):
            # solve matrix equation
            phiOld = np.linalg.solve(A, np.dot(B,phiOld))
            
            phi_hov[:,n]=phiOld
            variance_phi = np.mean((phiOld - np.mean(phiOld))**2)
            variance.append(variance_phi)
            mass.append(np.sum(phiOld)*dx)
                
        phi=phiOld
        
    
    # shift analytical solution forwards
    shift = int(np.round(u * dt * nt / dx))
    phi_exact = np.roll(phi0, shift)
    
    # compute error
    l2_error = np.sqrt(np.sum((phi - phi_exact)**2) * dx)
    
    #print("Relative change in mass",(mass[-1]-mass[0])/mass[0],len(mass))
    #print("Relative change in variance:", (variance[-1]-variance[0])/variance[0])
    return l2_error,phi_hov,mass,variance

def convergence_test(scheme, alpha_value=0):
    '''
    Tests for order of accuracy and convergence

    Parameters
    ----------
    scheme (string): Name of the scheme
    alpha_value (float): Asselin coefficient The default is 0.


    Returns
    -------
    errors (array of floats): array of errors for each of the tested resolutions
    order_est (float): estimated order of accuracy of the scheme

    '''
    # time and spatial resolutions
    nx_list = np.array([800, 1600, 3200, 6400])
    nt_list = 2*nx_list
    
    errors = np.zeros(len(nx_list))
    
    # compute errors for each of the four schemes
    
    for index, nx in enumerate(nx_list):
        if scheme == 'CTCS':
            l2_err = advect(1, nx, nt_list[index], 1,
                            scheme='CTCS',
                            alpha=alpha_value, use_square=False,)
        elif scheme == 'FTBS':
            l2_err = advect(1, nx, nt_list[index], 1,
                            scheme='FTBS',
                            alpha=0.0, use_square=False)
        elif scheme == 'SL':
            
            nt_list = np.array([600,1200,2400,4800])
            l2_err = advect(1, nx, nt_list[index], 1,
                            scheme='SL',
                            alpha=0, use_square=False)
        elif scheme == 'CN':
            # coarser resolution for CN
            nx_list = np.array([100, 200, 400, 800])
            nt_list = np.array([200,400,800,1600])
            l2_err = advect(1, nx_list[index], nt_list[index], 1,
                            scheme='CN',
                            alpha=alpha_value, use_square=False)
     
        
        
        errors[index] = l2_err[0]

 
    dx_list = 1.0 / nx_list
    
    # linear fit to error/resolution
    coeffs = np.polyfit(np.log(dx_list), np.log(errors), 1)
    order_est = coeffs[0]

    
    print(f"\nEstimated order of accuracy for {scheme}: {abs(order_est):.4f}")
    return errors,order_est


