import scipy
from matplotlib import pyplot as plt
import jax.numpy as jnp


grid = jnp.linspace(0, 10, 10)[None,:]

ts = jnp.arange(15)[:,None]

omegas = [3, 8]
funcs = []


for omega in omegas:

    funcs.append(lambda t, w= omega: jnp.cos(w*t*grid))


x = lambda t: sum(f(t) for f in funcs)


plt.plot(grid[0],x(1)[0])
plt.show()