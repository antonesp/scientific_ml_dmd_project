import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mr_dmd.DMD_funcs import mrDMD
from mr_dmd.overlapping_window_mrDMD import OmrDMD

def DMD(X, Xprime, r, energy_threshold=0.999):
    U, Sigma, VT = jnp.linalg.svd(X, full_matrices=False)  # Step 1
    # Choose r adaptively based on singular value energy (prevents nans from forming)
    total_energy = jnp.sum(Sigma**2)
    cumulative_energy = jnp.cumsum(Sigma**2) / total_energy
    r_adaptive = int(jnp.searchsorted(cumulative_energy, energy_threshold)) + 1
    r_adaptive = min(r, r_adaptive, len(Sigma))  # never exceed requested r

    Ur = U[:, :r_adaptive]
    Sigmar = jnp.diag(Sigma[:r_adaptive])

    Vr = VT[:r_adaptive, :].conj().T

    Atilde = Ur.conj().T @ Xprime @ Vr @ jnp.linalg.inv(Sigmar)

    Lambda, W = jnp.linalg.eig(Atilde)

    # Use the standard DMD mode definition
    Phi = Xprime @ Vr @ jnp.linalg.inv(Sigmar) @ W

    # SIMPLE AMPLITUDE CALCULATION
    # b represents the modal coordinates at t=0
    b = jnp.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]

    return Phi, Lambda, b


t_steps = 128
n_steps = 20
r = 30
t_max = 128


dt = t_max / t_steps


def f(start, stop, t):
    # This uses JAX logic to return 0 or 1 without Python if/else
    return jnp.where((t >= start) & (t <= stop), 1.0, 0.0)


ts = jnp.linspace(0, t_max, t_steps)

g_1 = lambda t: jnp.full_like(t, 0.2)
g_2 = lambda t: 2 * jnp.cos(1 / (64) * 2 * jnp.pi * t)

g_3 = lambda t: 2 * jnp.sin(1 / 32 * jnp.pi * 2 * t) * f(0, 64, t)
g_4 = lambda t: 2 * jnp.sin(1 / 8 * t * jnp.pi * 2) * f(64, 96, t)


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
    # + g_3(t)[:, None, None] * psi3
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
L = 6
r = 20

Phis_DMD, Lambda, b = DMD(X_hankel, Xprime_hankel, r, energy_threshold=2)
Phis, func, time_funcs = mrDMD(X_hankel, Xprime_hankel, M, L, f, dt, ts)

mrDMD_reconstruct = jnp.zeros_like(data_flat)

for i in range(t_steps):
    X_at_step = func(ts[i])[:spatial_size]
    mrDMD_reconstruct = mrDMD_reconstruct.at[:, i].set(X_at_step.ravel())

print(func(118))
print(func(119))
print(func(127))
# print(mrDMD_reconstruct[0,0:17])
k = jnp.arange(t_steps)
v_lambda = Lambda[:, None] ** k
dynamics = b[:, None] * v_lambda

X_rec = jnp.dot(Phis_DMD, dynamics)
X_rec_real = jnp.real(X_rec)


print(X_rec_real.shape)

# print("Normal DMD square error ",jnp.mean((X_rec_real[:spatial_size]-data_flat)**2, axis=0))
# print("mrDMD square errpor ", jnp.mean((mrDMD_reconstruct - data_flat)**2, axis= 0))
error_DMD = jnp.mean((X_rec_real[:spatial_size] - data_flat) ** 2, axis=0)
error_mrDMD = jnp.mean((mrDMD_reconstruct - data_flat) ** 2, axis=0)


plt.plot(ts, error_DMD, label="Normal DMD")
plt.plot(ts, jnp.mean((mrDMD_reconstruct - data_flat) ** 2, axis=0), label="mrDMD")
plt.legend()
plt.show()


# Animation of mrDMD_reconstruct frames
fig2, ax2 = plt.subplots()
frame0 = jnp.real(mrDMD_reconstruct[:, 0]).reshape(len(ys), len(xs))
im = ax2.imshow(
    np.asarray(frame0).T,
    cmap="viridis",
    origin="lower",
    vmin=jnp.min(mrDMD_reconstruct),
    vmax=jnp.max(mrDMD_reconstruct),
)
ax2.set_title("mrDMD reconstruction")


def update(frame):
    data = jnp.real(mrDMD_reconstruct[:, frame]).reshape(len(ys), len(xs))
    im.set_data(np.asarray(data).T)
    ax2.set_xlabel(f"Time step: {frame}")
    return (im,)


ani = animation.FuncAnimation(fig2, update, frames=t_steps, interval=100, blit=True)
plt.show()


# all_trajectories = []
# for tf in time_funcs:
#     # Evaluate the lambda function for the entire time array
#     # tf(t_eval) returns (r_low, t_steps)
#     traj = tf(ts)

#     all_trajectories.append(traj)

# all_modes_matrix = jnp.vstack(all_trajectories)

# n_modes = all_modes_matrix.shape[0]
# fig, axes = plt.subplots(n_modes, 1, figsize=(10, 2.5 * n_modes), sharex=True)
# if n_modes == 1:
#     axes = [axes]

# for i, ax in enumerate(axes):
#     ax.plot(ts, all_modes_matrix.real[i], alpha=0.7)
#     ax.set_title(f"Mode {i + 1}")
#     ax.set_ylabel("Amplitude")


# axes[-1].set_xlabel("Time")
# plt.tight_layout()
# # plt.show()

# k = jnp.arange(len(ts))  # [0, 1, 2, ..., 99]

# all_time_functions = b[:, None] * (Lambda[:, None] ** k)


# reconstructed_g_all = jnp.real(all_time_functions)


# fig, axes = plt.subplots(reconstructed_g_all.shape[0], 1, figsize=(10, 2.5 * reconstructed_g_all.shape[0]), sharex=True)
# if reconstructed_g_all.shape[0] == 1:
#     axes = [axes]

# for i, ax in enumerate(axes):
#     ax.plot(np.asarray(k), np.asarray(reconstructed_g_all[i]))
#     ax.set_title(f"Reconstructed mode {i + 1}")
#     ax.set_ylabel("Amplitude")

# axes[-1].set_xlabel("Time index")
# plt.tight_layout()
# # plt.show()


# num_phis = len(Phis)
# fig, ax = plt.subplots(1, num_phis, figsize=(5 * num_phis, 5), squeeze=False)

# for i, phi in enumerate(Phis):
#     phi_grid = jnp.real(phi[:spatial_size, 0]).reshape(len(ys), len(xs))
#     im = ax[0, i].contourf(X, Y, phi_grid.T, levels=20, cmap="hot")
#     ax[0, i].set_title(f"Phi {i + 1}")
#     ax[0, i].set_xlabel("x")
#     ax[0, i].set_ylabel("y")

# plt.tight_layout()
# # plt.show()


# num_phis_dmd = Phis_DMD.shape[1]
# fig, ax = plt.subplots(1, num_phis_dmd, figsize=(5 * num_phis_dmd, 5), squeeze=False)

# for i in range(num_phis_dmd):
#     phi_grid = jnp.real(Phis_DMD[:spatial_size, i]).reshape(len(ys), len(xs))
#     im = ax[0, i].contourf(X, Y, phi_grid.T, levels=20, cmap="hot")
#     ax[0, i].set_title(f"DMD Phi {i + 1}")
#     ax[0, i].set_xlabel("x")
#     ax[0, i].set_ylabel("y")

# plt.tight_layout()
# # plt.show()
