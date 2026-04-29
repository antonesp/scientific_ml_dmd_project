import scipy
from matplotlib import pyplot as plt
import jax.numpy as jnp

import jax.random as random

from mr_dmd.helper_functions import sum_of_sines

t_steps = 200
n_steps = 20
r = 30
t_max = 20


# g = lambda x,t : 1*jnp.cos(x + t)
seed = 5
g = sum_of_sines(seed, 10)

def f(start, stop, t):
    if (t < start) or (t > stop): return 0
    else: return 1

x = jnp.linspace(0, 2*jnp.pi, n_steps)
x_precise = jnp.linspace(0, 2*jnp.pi, 500)
t = jnp.linspace(0, t_max, t_steps)



raw = (g(x[:, None],t[None, :]))
