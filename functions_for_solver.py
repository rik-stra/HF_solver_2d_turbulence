import numpy as np
import h5py
import sys
import torch
from filters import Filters

######################
# SOLVER SUBROUTINES #
######################

#pseudo-spectral technique to solve for Fourier coefs of Jacobian
def compute_VgradW_hat(w_hat_n, psi_hat_n, P, kx, ky):
    """Computes u_n*w_x_n + v_n*w_y_n using the pseudo-spectral method

    Args:
        w_hat_n (complex): fourier coef of vorticity
        P (_type_): filter
        kx (_type_): first entry wavenumer vector
        ky (_type_): second entry wavenumber vector
        k_squared_no_zero (_type_): 

    Returns:
        _type_: filtered fourier coefficients for VgradW_hat
    """
    u_n = torch.fft.ifft2(-ky*psi_hat_n).real
    v_n = torch.fft.ifft2(kx*psi_hat_n).real
    #compute jacobian in physical space Non-conservative
    w_x_n = torch.fft.ifft2(kx*w_hat_n).real
    w_y_n = torch.fft.ifft2(ky*w_hat_n).real
    VgradW_n_nc = u_n*w_x_n + v_n*w_y_n
    #return to spectral space
    VgradW_hat_n_nc = torch.fft.fft2(VgradW_n_nc)
    VgradW_hat_n_nc *= P
    
    #compute jacobian conservative
    w_n = torch.fft.ifft2(w_hat_n).real
    VgradW_hat_n_c = kx*torch.fft.fft2(u_n*w_n)+ky* torch.fft.fft2(v_n*w_n)
    VgradW_hat_n_c *= P

    VgradW_hat_n = 0.5* (VgradW_hat_n_c+VgradW_hat_n_nc)

    return VgradW_hat_n



#get Fourier coefficient of the vorticity at next (n+1) time step
def get_w_hat_np1(method,dt,mu,nu,w_hat_n, psi_hat_n, w_hat_nm1, VgradW_hat_nm1, P, norm_factor, kx, ky, k_squared, F_hat, sgs_hat = 0.0):
    
    #compute jacobian
    VgradW_hat_n = compute_VgradW_hat(w_hat_n, psi_hat_n, P, kx, ky)
    
    if method == "AB/BDI2":
        #solve for next time step according to AB/BDI2 scheme
        w_hat_np1 = norm_factor*(2.0/dt*w_hat_n - 1.0/(2.0*dt)*w_hat_nm1 - \
                                2.0*VgradW_hat_n + VgradW_hat_nm1 + mu*F_hat - sgs_hat)
    elif method == "AB/CN":
        #solve for next time step according to AB/CN scheme
        w_hat_np1 = norm_factor*P*(2/dt*w_hat_n + nu*k_squared*w_hat_n - (3*VgradW_hat_n - VgradW_hat_nm1) + mu*(2*F_hat-w_hat_n)- sgs_hat)
    
    return w_hat_np1, VgradW_hat_n


#return the fourier coefs of the stream function
def get_psi_hat(w_hat_n, k_squared_no_zero):

    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0

    return psi_hat_n


def compute_VgradW_hat_np(w_hat_n, psi_hat_n, P, kx, ky):
    """Computes u_n*w_x_n + v_n*w_y_n using the pseudo-spectral method

    Args:
        w_hat_n (complex): fourier coef of vorticity
        P (_type_): filter
        kx (_type_): first entry wavenumer vector
        ky (_type_): second entry wavenumber vector
        k_squared_no_zero (_type_): 

    Returns:
        _type_: filtered fourier coefficients for VgradW_hat
    """
    u_n = np.fft.ifft2(-ky*psi_hat_n).real
    v_n = np.fft.ifft2(kx*psi_hat_n).real
    #compute jacobian in physical space Non-conservative
    w_x_n = np.fft.ifft2(kx*w_hat_n).real
    w_y_n = np.fft.ifft2(ky*w_hat_n).real
    VgradW_n_nc = u_n*w_x_n + v_n*w_y_n
    #return to spectral space
    VgradW_hat_n_nc = np.fft.fft2(VgradW_n_nc)
    VgradW_hat_n_nc *= P
    
    #compute jacobian conservative
    w_n = np.fft.ifft2(w_hat_n).real
    VgradW_hat_n_c = kx*np.fft.fft2(u_n*w_n)+ky* np.fft.fft2(v_n*w_n)
    VgradW_hat_n_c *= P

    VgradW_hat_n = 0.5* (VgradW_hat_n_c+VgradW_hat_n_nc)

    return VgradW_hat_n
