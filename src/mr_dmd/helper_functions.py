from scipy.stats import norm
from scipy.stats import t as t_dist
import numpy as np
import jax.random as random
import jax.numpy as jnp
import pywt


def Schroeder_sines(n, x, t,seed = 1, amplitude = 0.1, max_power = 100):
    # Frequencies and Amplitudes (Uniform for flat power spectrum)
    
    omega = amplitude * norm.rvs(0,4,random_state= seed*2, size= n).reshape(n, 1)
    A = np.ones((n, 1)) 
    phi_0 = norm.rvs(0,4,random_state= seed)
    # Schroeder Phase Formula
    k = np.arange(n).reshape(n, 1)
    phi = phi_0 + (np.pi * k**2) / n


    # Vectorized sum
    u = np.sum(A * np.sin(omega * t + phi*x), axis=0)

    u = u - np.min(u)
    u_max = np.max(u)

    if u_max > 0:
        u = u / u_max
    return u*max_power



def sum_of_sines(seed, n):

    key = random.key(seed)
    key, subkey = random.split(key)

    omegas = 5*random.normal(subkey, shape=(n,))
    amplitudes = random.normal(key, shape=(n,))

    funcs = []

    for i in range(n):

        funcs.append(lambda t, x, w= omegas[i], z = amplitudes[i]: z*jnp.sin(w*t + x))


    f = lambda t, x: sum(f(t,x) for f in funcs)
    return f

def indicator(start, stop, t):
    return 1.0 if start <= t <= stop else 0.0

def make_wavelet_window(wavelet='morl', level =4):
    wav = pywt.ContinuousWavelet(wavelet)
    psi, x = wav.wavefun(level=level)  # computed once
    psi_real = np.real(psi)
    psi_norm = psi_real / np.max(np.abs(psi_real))

    def window(start, stop, t):
        center = (start + stop) / 2
        width  = (stop - start)
        t_normalized = 6 * (t - center) / (width + 1e-10)
        val = psi(t_normalized)
        print(f"t={t:.3f}, center={center:.3f}, width={width:.3f}, t_norm={t_normalized:.3f}, val={val:.6f}")
        val_norm = val / psi(0.0)
        return float(val_norm)
    
    return window