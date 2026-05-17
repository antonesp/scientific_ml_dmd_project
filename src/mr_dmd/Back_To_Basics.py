import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mr_dmd.DMD_funcs import mrDMD, DMD
from mr_dmd.overlapping_window_mrDMD import OmrDMD


t_steps = 128
n_steps = 20
t_max = 128


dt = t_max / t_steps

def f(start, stop, t):
    # This uses JAX logic to return 0 or 1 without Python if/else
    return jnp.where((t >= start) & (t <= stop), 1.0, 0.0)


ts = jnp.linspace(0, t_max, t_steps)

g_1 = lambda t: jnp.full_like(t, 0.2)
g_2 = lambda t: 2 * jnp.cos(1 / 64 * 2 * jnp.pi * t)

g_3 = lambda t: 2 * jnp.sin(1 / 32 * jnp.pi * 2 * t) * f(0, 64, t)
g_4 = lambda t: 2 * jnp.sin(1 / 16 * jnp.pi * 2 * t) * f(64, 96, t)


g_funcs = [g_1, g_2, g_3, g_4]

xs = jnp.linspace(-20, 20, 200)
ys = jnp.linspace(-20, 20, 200)

X, Y = jnp.meshgrid(xs, ys)


k_high = 10
k_mid = 0.8
k_low = 0.3


psi1 = np.abs(
    np.sin(k_high * X) * np.sin(k_high * Y)
    + np.sin(k_low * X) * np.sin(k_low * Y)
    + np.sin(k_mid * X)
    + np.sin(k_mid * Y)
)

x0, y0 = -10, 10
sigma = 4
psi2 = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))

x0, y0 = 5, 5
sigma_x, sigma_y = 12, 4
psi3 = np.exp(-((X - x0) ** 2 / (2 * sigma_x**2) + (Y - y0) ** 2 / (2 * sigma_y**2)))


x_offset = 8
sigma_lobe = 5
psi4 = np.exp(-((X + x_offset) ** 2 + Y**2) / (2 * sigma_lobe**2)) + np.exp(
    -((X - x_offset) ** 2 + Y**2) / (2 * sigma_lobe**2)
)


final_func = lambda t: (
    g_1(t)[:, None, None] * psi1
    + g_2(t)[:, None, None] * psi2
    + g_3(t)[:, None, None] * psi3
    + g_4(t)[:, None, None] * psi4
)

raw = final_func(ts)

data_flat = raw.T.reshape((raw.shape[1] * raw.shape[2], raw.shape[0]))
data_flat = data_flat

input = data_flat[:, :-1]
output = data_flat[:, 1:]


def create_hankel_matrix(data, rows):
    # data: (Space, Time)
    # rows: number of delay taps (try 2 or 10)
    return jnp.vstack([data[:, i : data.shape[1] - rows + i + 1] for i in range(rows)])


H = create_hankel_matrix(data_flat, rows=10)  # 2 is enough for a simple cosine
print("Shape of Hankel matrix", H.shape)
print("Shape of data flat", data_flat.shape)

X_hankel = H[:, :-1]
Xprime_hankel = H[:, 1:]

spatial_size = len(xs) * len(ys)

M = 6
L = 7
r = 30


Phis, func, time_funcs = mrDMD(input, output, M, L, f, dt, ts)
# r = len(Phis)*4

Phis_DMD, Lambda, b = DMD(input, output, r, energy_threshold=2)

mrDMD_reconstruct = jnp.zeros_like(data_flat)

for i in range(t_steps):
    X_at_step = func(ts[i])[:spatial_size]
    mrDMD_reconstruct = mrDMD_reconstruct.at[:, i].set(X_at_step.ravel())


k = jnp.arange(t_steps)
v_lambda = Lambda[:, None] ** k
dynamics = b[:, None] * v_lambda

X_rec = jnp.dot(Phis_DMD, dynamics)
X_rec_real = jnp.real(X_rec)


error_DMD = jnp.mean((X_rec_real[:spatial_size] - data_flat) ** 2, axis=0)
error_mrDMD = jnp.mean((mrDMD_reconstruct - data_flat) ** 2, axis=0)


plt.plot(ts[:-12], error_DMD[:-12], label="Normal DMD")
plt.plot(ts[:-12], jnp.mean((mrDMD_reconstruct - data_flat) ** 2, axis=0)[:-12], label="mrDMD")
plt.legend()
plt.show()


DMD_reconstruct = X_rec_real[:spatial_size,:]


# Animation of mrDMD_reconstruct frames
fig2, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
frame0_mr = jnp.real(mrDMD_reconstruct[:, 0]).reshape(len(ys), len(xs))
frame0_dmd = jnp.real(DMD_reconstruct[:, 0]).reshape(len(ys), len(xs))
frame0_true = jnp.real(data_flat[:, 0]).reshape(len(ys), len(xs))
vmin = float(jnp.minimum(jnp.minimum(jnp.min(mrDMD_reconstruct), jnp.min(DMD_reconstruct)), jnp.min(data_flat)))
vmax = float(jnp.maximum(jnp.maximum(jnp.max(mrDMD_reconstruct), jnp.max(DMD_reconstruct)), jnp.max(data_flat)))
im_dmd = ax2.imshow(
    np.asarray(frame0_dmd).T,
    cmap="jet",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
)
im_mr = ax3.imshow(
    np.asarray(frame0_mr).T,
    cmap="jet",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
)
im_true = ax4.imshow(
    np.asarray(frame0_true).T,
    cmap="jet",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
)
ax2.set_title("DMD reconstruction")
ax3.set_title("mrDMD reconstruction")
ax4.set_title("True Data")


def update(frame):
    data_mr = jnp.real(mrDMD_reconstruct[:, frame]).reshape(len(ys), len(xs))
    data_dmd = jnp.real(DMD_reconstruct[:, frame]).reshape(len(ys), len(xs))
    data_true = jnp.real(data_flat[:, frame]).reshape(len(ys), len(xs))
    im_mr.set_data(np.asarray(data_mr).T)
    im_dmd.set_data(np.asarray(data_dmd).T)
    im_true.set_data(np.asarray(data_true).T)
    return (im_mr, im_dmd, im_true)


ani = animation.FuncAnimation(fig2, update, frames=t_steps-12, interval=100, blit=True)

ani.save("images/mr_dmd_comparison.gif", writer="pillow", fps=10)
plt.show()

