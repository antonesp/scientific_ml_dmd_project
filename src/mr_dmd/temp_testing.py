from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import jax.numpy as jnp
from mr_dmd.DMD_funcs import mrDMD, DMD
import numpy as np

t_steps = 500
n_steps = 20
r = 30
t_max = 128


# Create the time functions assosciated with each mode

def f(start, stop, t):
    # This uses JAX logic to return 0 or 1 without Python if/else
    return jnp.where((t >= start) & (t <= stop), 1.0, 0.0)


ts = jnp.linspace(0, t_max, t_steps)

g_1 = lambda t: jnp.full_like(t, 0.1)
g_2 = lambda t: 2*jnp.cos(1/(64)*2*jnp.pi*t)

g_3 = lambda t: 2*jnp.sin(1/16*jnp.pi*2*t) * f(0,64,t)
g_4 = lambda t: 2*jnp.sin(1/4*t*jnp.pi*2) * f(64,96,t)


g_funcs = [g_1, g_2, g_3, g_4]


# Create the grid and modes for the final function
xs = jnp.linspace(-20,20,200)
ys = jnp.linspace(-20,20,200)

X, Y = jnp.meshgrid(xs, ys)


k_high = 10
k_mid = 0.8
k_low = 0.3


psi1 = np.abs(np.sin(k_high * X) * np.sin(k_high * Y) + 
                np.sin(k_low * X) * np.sin(k_low * Y)
                + np.sin(k_mid*X) + np.sin(k_mid*Y))

x0, y0 = -10, 10
sigma = 4
psi2 = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

x0, y0 = 5, 5
sigma_x, sigma_y = 12, 4
psi3 = np.exp(-((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2)))


x_offset = 8
sigma_lobe = 5
psi4 = (np.exp(-((X + x_offset)**2 + Y**2) / (2 * sigma_lobe**2)) + 
        np.exp(-((X - x_offset)**2 + Y**2) / (2 * sigma_lobe**2)))


# Plot individual modes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
psi_modes = [psi1, psi2, psi3, psi4]
titles = ["Mode 1: Oscillatory", "Mode 2: Isotropic Gaussian", "Mode 3: Anisotropic Gaussian", "Mode 4: Bimodal"]

for ax, psi, title in zip(axes.flat, psi_modes, titles):
    im = ax.contourf(X, Y, psi, levels=20, cmap='hot')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()


# Combine modes and frequencies together
final_func = lambda t: (
        g_1(t)[:, None, None] * psi1
        + g_2(t)[:, None, None] * psi2
        + g_3(t)[:, None, None] * psi3
        + g_4(t)[:, None, None] * psi4
)

raw = final_func(ts)





# Flatten the into the shape (t, x, y) -> (x*y, t)
data_flat = raw.T.reshape((raw.shape[1]*raw.shape[2], raw.shape[0]))
data_flat = data_flat - jnp.mean(data_flat, axis= 0)
dt = t_max/t_steps

input = data_flat[:,:-1]
target = data_flat[:,1:]

M = 8
L = 3

# Generate the mrDMD results
# Phis, func, time_funcs = mrDMD(input, target, M, L, f, dt, ts)
Phis_DMD, Lambda, b = DMD(input,target, 16, energy_threshold=2)

print(Lambda)
omegas_DMD = jnp.log(Lambda)/dt


DMD_t_funcs = lambda t: b[:, None] * jnp.exp(omegas_DMD[:, None] * t[None, :])


X_dmd = Phis_DMD @ (b[:, None] * jnp.exp(omegas_DMD[:, None] * ts[None, :]))

# 2. Project the reconstruction back onto your GROUND TRUTH spatial modes
# Assuming psi_truth is shape (4, Space)
# This gives you exactly 4 time functions, even if DMD rank was 10 or 20
# psi_truth should be shape (4, Number_of_Pixels)
psi_truth = jnp.stack([psi1.flatten(), psi2.flatten(), psi3.flatten(), psi4.flatten()])
g_dmd_reconstructed = psi_truth @ X_dmd 




# 3. Plot them against the originals
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i in range(4):
    ax = axes.flat[i]
    ax.plot(ts, g_funcs[i](ts), 'k--', label='True')
    ax.plot(ts, jnp.real(g_dmd_reconstructed[i]), label='DMD Recon')
    ax.set_title(f'Mode {i+1}')
    ax.set_xlabel('t')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
# print(jnp.imag(Lambda))
# print("PhiDMD shape", Phis_DMD.shape)

# b = jnp.linalg.lstsq(Phis_DMD, data_flat[:, 0])[0] 

# # 3. Use your exponential formula (the "DMD Physics")
# omegas_DMD = jnp.log(Lambda) / dt
# time_function_values = b[:, None] * jnp.exp(omegas_DMD[:, None] * ts[None, :])

# # 4. Take the Real part (since your g(t) are real)
# g_dmd = jnp.real(time_function_values)

# # Plot g_reconstructed
# num_modes = g_dmd.shape[0]
# fig, ax = plt.subplots(num_modes, 1, figsize=(10, 3 * num_modes), squeeze=False)

# for i in range(num_modes):
#     ax[i, 0].plot(np.asarray(ts), np.asarray(g_dmd[i]), label=f"Mode {i + 1}")
#     ax[i, 0].set_title(f"Reconstructed Time Function {i + 1}")
#     ax[i, 0].set_xlabel("t")
#     ax[i, 0].set_ylabel("Amplitude")
#     ax[i, 0].grid(True)
#     ax[i, 0].legend()

# plt.tight_layout()
# plt.show()



# print("output shape,", y_vals.shape)

# num_phis = len(Phis)
# fig, ax = plt.subplots(1, num_phis, figsize=(5 * num_phis, 5), squeeze=False)

# for i, phi in enumerate(Phis):
#     phi_grid = jnp.real(phi[:, 0]).reshape(len(ys), len(xs))
#     im = ax[0, i].contourf(X, Y, phi_grid.T, levels=20, cmap="hot")
#     ax[0, i].set_title(f"Phi {i + 1}")
#     ax[0, i].set_xlabel("x")
#     ax[0, i].set_ylabel("y")

# plt.tight_layout()
# plt.show()




# num_phis_dmd = Phis_DMD.shape[1]
# fig, ax = plt.subplots(1, num_phis_dmd, figsize=(5 * num_phis_dmd, 5), squeeze=False)

# for i in range(num_phis_dmd):
#     phi_grid = jnp.real(Phis_DMD[:, i]).reshape(len(ys), len(xs))
#     im = ax[0, i].contourf(X, Y, phi_grid.T, levels=20, cmap="hot")
#     ax[0, i].set_title(f"DMD Phi {i + 1}")
#     ax[0, i].set_xlabel("x")
#     ax[0, i].set_ylabel("y")

# plt.tight_layout()
# plt.show()

# #### We now evaluate the time functions for each mode
# # We create them for normal DMD

# # omega DMD

# print("before y")
# y_vals = jnp.array([time_funcs[0](t) for t in ts])
# print("after y")
# omegas_DMD = jnp.log(Lambda)/dt


# DMD_t_funcs = lambda t: b[:, None] * jnp.exp(omegas_DMD[:, None] * t[None, :])

# time_function_values = DMD_t_funcs(ts)

# num_time_funcs = time_function_values.shape[0]
# fig_time, ax_time = plt.subplots(num_time_funcs, 1, figsize=(10, 3 * num_time_funcs), squeeze=False)

# for i in range(num_time_funcs):
#     ax_time[i, 0].plot(np.asarray(ts), np.asarray(time_function_values[i]), label = "DMD", color = "r")
#     ax_time[i, 0].set_title(f"DMD Time Function {i + 1}")
#     ax_time[i, 0].set_xlabel("t")
#     ax_time[i, 0].set_ylabel("Amplitude")
#     ax_time[i, 0].grid(True)
#     ax_time[i,0].plot(np.asanyarray(ts),y_vals[:, i], label = "mrDMD", linestyle = "--", color = "g")
    
#     ax_time[i,0].legend()
# plt.tight_layout()

# plt.show()




