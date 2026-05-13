import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mr_dmd.DMD_funcs import mrDMD

def DMD(X,Xprime, r, energy_threshold=0.999):
    U, Sigma, VT = jnp.linalg.svd(X,full_matrices=False) # Step 1
    # Choose r adaptively based on singular value energy (prevents nans from forming)
    total_energy = jnp.sum(Sigma**2)
    cumulative_energy = jnp.cumsum(Sigma**2) / total_energy
    r_adaptive = int(jnp.searchsorted(cumulative_energy, energy_threshold)) + 1
    r_adaptive = min(r, r_adaptive, len(Sigma))  # never exceed requested r

    Ur = U[:,:r_adaptive]
    Sigmar = jnp.diag(Sigma[:r_adaptive])

    Vr = VT[:r_adaptive,:].conj().T

    Atilde = Ur.conj().T @ Xprime @ Vr @  jnp.linalg.inv(Sigmar)
    
    Lambda, W = jnp.linalg.eig(Atilde)
    
    # Use the standard DMD mode definition
    Phi = Xprime @ Vr @ jnp.linalg.inv(Sigmar) @ W
    
    # SIMPLE AMPLITUDE CALCULATION
    # b represents the modal coordinates at t=0
    b = jnp.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]

    return Phi, Lambda, b


t_steps = 500
n_steps = 20
r = 30
t_max = 128


dt =  t_max / t_steps


def f(start, stop, t):
    # This uses JAX logic to return 0 or 1 without Python if/else
    return jnp.where((t >= start) & (t <= stop), 1.0, 0.0)


ts = jnp.linspace(0, t_max, t_steps)

g_1 = lambda t: jnp.full_like(t, 0.1)
g_2 = lambda t: 2*jnp.cos(1/(64)*2*jnp.pi*t)

g_3 = lambda t: 2*jnp.sin(1/16*jnp.pi*2*t) * f(0,64,t)
g_4 = lambda t: 2*jnp.sin(1/4*t*jnp.pi*2) * f(64,96,t)


g_funcs = [g_1, g_2, g_3, g_4]

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


final_func = lambda t: (
        g_1(t)[:, None, None] * psi1
        + g_2(t)[:, None, None] * psi2
        + g_3(t)[:, None, None] * psi3
        + g_4(t)[:, None, None] * psi4
)

raw = final_func(ts)

data_flat = raw.T.reshape((raw.shape[1]*raw.shape[2], raw.shape[0]))
data_flat = data_flat

# input = data_flat[:, :-1]
# output = data_flat[:,  1:]

def create_hankel_matrix(data, rows):
    # data: (Space, Time)
    # rows: number of delay taps (try 2 or 10)
    return jnp.vstack([data[:, i:data.shape[1]-rows+i+1] for i in range(rows)])


H = create_hankel_matrix(data_flat, rows=10) # 2 is enough for a simple cosine
X_hankel = H[:, :-1]
Xprime_hankel = H[:, 1:]


M = 8
L = 3

Phis_DMD, Lambda, b = DMD(X_hankel, Xprime_hankel, 8, energy_threshold=2)
Phis, func, time_funcs = mrDMD(X_hankel, Xprime_hankel, M, L, f, dt, ts)


all_trajectories = []
for tf in time_funcs:
    # Evaluate the lambda function for the entire time array
    # tf(t_eval) returns (r_low, t_steps)
    traj = tf(ts)
    all_trajectories.append(traj)

all_modes_matrix = jnp.vstack(all_trajectories)

plt.figure(figsize=(10, 6))
# Plot the real part of all discovered trajectories
# We use .real because DMD produces complex conjugate pairs for oscillations
plt.plot(ts, all_modes_matrix.real.T, alpha=0.5)
plt.title("All Rediscovered Temporal Dynamics (mrDMD)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

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
# plt.show()


