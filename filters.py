from typing import Any
import numpy as np
from scipy import stats
import torch
from math import ceil, floor

def get_dealiasing_filter(N,remove_alias,rfft = True, filter_size=None): # square filter!
    if filter_size == None: filter_size = N
    
    if rfft: k = np.fft.rfftfreq(filter_size).reshape(1,-1)*filter_size
    else: k = np.fft.fftfreq(filter_size).reshape(1,-1)*filter_size
    l = np.fft.fftfreq(filter_size).reshape(-1,1)*filter_size
    P = np.ones([l.shape[0], k.shape[1]],dtype=bool)
    if remove_alias:
        cutoff = int((N-1)/3)
        P = np.where(np.abs(k)<=cutoff,P,0)
        P = np.where(np.abs(l)<=cutoff,P,0)
    else:
        P = np.where(np.abs(k)<np.ceil(N/2),P,0)
        P = np.where(np.abs(l)<np.ceil(N/2),P,0)
        
    return P


def _get_ksquared(kx,ky):
    k_squared = kx**2 + ky**2
    k_squared_no_zero = np.copy(k_squared)
    k_squared_no_zero[0,0] = 1.0
    return k_squared, k_squared_no_zero

class Grids:
    def __init__(self, N_HF, N_HF_resolved):
        self.N_HF = N_HF
        self.N_HF_resolved = N_HF_resolved
        
        axis_HF = np.linspace(0, 2.0*np.pi, N_HF, endpoint=False)
        self.x_HF , self.y_HF = np.meshgrid(axis_HF, axis_HF)
        self.k_x_HF = 1j * np.fft.fftfreq(N_HF).reshape(-1,1)*N_HF
        self.k_y_HF = 1j * np.fft.fftfreq(N_HF).reshape(1,-1)*N_HF

        self.k_squared_HF, self.k_squared_nonzero_HF = _get_ksquared(self.k_x_HF, self.k_y_HF)

        


class Filters:
    def __init__(self,grid:Grids, remove_alias:bool):

        self.N_HF = grid.N_HF
        self.N_HF_resolved = grid.N_HF_resolved
        self.remove_alias = remove_alias
        
        self.Ncutoff_HF=int(grid.N_HF/(2+remove_alias))
        self.P_HF = get_dealiasing_filter(grid.N_HF,remove_alias,rfft=False)

    
    def filter_HF2resolved(self, x):
        if not self.remove_alias:
            return x
        else:
            return x[self.P_HF==1].reshape((self.N_HF_resolved,self.N_HF_resolved))*(self.N_HF_resolved/self.N_HF)**2
    
    def fill_LF_with_resolved(self, LF_array, resolved_array):
        if not self.remove_alias:
            LF_array = resolved_array
        else:
            a = ceil(self.N_LF_resolved/2)
            b = -floor(self.N_LF_resolved/2)
            LF_array[:a,:a] = resolved_array[:a,:a]
            LF_array[:a,b:] = resolved_array[:a,b:]
            LF_array[b:,:a] = resolved_array[b:,:a]
            LF_array[b:,b:] = resolved_array[b:,b:]
        return LF_array*(self.N_LF/self.N_LF_resolved)**2
