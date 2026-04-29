from scipy.stats import norm
from scipy.stats import t as t_dist
import numpy as np


def Schroeder_sines(n, t,seed = 1, amplitude = 0.1, max_power = 100):
    # Frequencies and Amplitudes (Uniform for flat power spectrum)
    
    omega = amplitude * norm.rvs(0,4,random_state= seed*2, size= n).reshape(n, 1)
    A = np.ones((n, 1)) 
    phi_0 = norm.rvs(0,4,random_state= seed)
    # Schroeder Phase Formula
    k = np.arange(n).reshape(n, 1)
    phi = phi_0 + (np.pi * k**2) / n


    # Vectorized sum
    u = np.sum(A * np.sin(omega * t + phi), axis=0)

    u = u - np.min(u)
    u_max = np.max(u)

    if u_max > 0:
        u = u / u_max
    return u*max_power