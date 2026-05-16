from scipy.stats import norm
import numpy as np
import jax.random as random
import jax.numpy as jnp
import pywt


def Schroeder_sines(n, x, t, seed=1, amplitude=0.1, max_power=100):
    # Frequencies and Amplitudes (Uniform for flat power spectrum)

    omega = amplitude * norm.rvs(0, 4, random_state=seed * 2, size=n).reshape(n, 1)
    A = np.ones((n, 1))
    phi_0 = norm.rvs(0, 4, random_state=seed)
    # Schroeder Phase Formula
    k = np.arange(n).reshape(n, 1)
    phi = phi_0 + (np.pi * k**2) / n

    # Vectorized sum
    u = np.sum(A * np.sin(omega * t + phi * x), axis=0)

    u = u - np.min(u)
    u_max = np.max(u)

    if u_max > 0:
        u = u / u_max
    return u * max_power


def sum_of_sines(seed, n):
    key = random.key(seed)
    key, subkey = random.split(key)

    omegas = 5 * random.normal(subkey, shape=(n,))
    amplitudes = random.normal(key, shape=(n,))

    funcs = []

    for i in range(n):
        funcs.append(lambda t, x, w=omegas[i], z=amplitudes[i]: z * jnp.sin(w * t + x))

    f = lambda t, x: sum(f(t, x) for f in funcs)
    return f


def indicator(start, stop, t):
    return 1.0 if start <= t <= stop else 0.0

def hann_window(start, stop):
    return lambda t, s=start, e=stop: jnp.where(
        (t >= s) & (t < e),
        0.5 * (1 - jnp.cos(2 * jnp.pi * (t - s) / (e - s))),
        0.0
    )