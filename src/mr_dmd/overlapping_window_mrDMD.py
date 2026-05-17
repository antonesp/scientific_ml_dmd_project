from matplotlib import pyplot as plt
import jax.numpy as jnp
from mr_dmd.helper_functions import hann_window

def DMD(X, Xprime, r, energy_threshold=0.999):
    if jnp.linalg.norm(X) < 1e-9:
        return None, None, None

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

    Lambda, W = jnp.linalg.eig(Atilde)  # Step 3
    Lambda = jnp.where(jnp.abs(Lambda) < 1e-12, 1e-12, Lambda)

    Phi = Xprime @ Vr @ jnp.linalg.inv(Sigmar) @ W
    b = jnp.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]  # Compute b as in paper

    return Phi, Lambda, b


def DMD_safe(X, Y, r, dt, energy_threshold):
    # 1. Energy check: if the bin is already empty, return None
    if jnp.linalg.norm(X) < 1e-5:
        return None, None, None

    # 2. Perform standard DMD
    Phi, Lambda, b = DMD(X, Y, r, energy_threshold)

    # 3. Handle Numerical Singularities
    eps = 1e-7
    Lambda_safe = jnp.where(
    jnp.abs(Lambda) < eps,
    eps * jnp.exp(1j * jnp.angle(Lambda)),
    Lambda,
)
    Lambda_clipped = jnp.where(jnp.abs(Lambda) < 1e-7, 1e-7, Lambda)  # Clip Lambda to prevent log(0)

    # 4. Convert to Omega and CLAMP
    omega = jnp.log(Lambda_safe) / dt
    omega = jnp.minimum(jnp.real(omega), 10.0) + 1j * jnp.imag(
        omega
    )  # Force Re(omega) <= 0 to prevent the explosion at t=64

    return Phi, b, omega


def OmrDMD(X, Y, M, L, f, dt, ts, energy_threshold=0.999, window_function = lambda start, stop, t: 0.5*jnp.ones_like(t)):
    """

    Overlapping Multi resolution DMD function:

    - X:  Datapoints at t_n (n_spacial, T)
    - Y:  Datapoints at t_n+1 (n_spacial, T)
    - M:  number of modes used in the first level when computing DMD
    - L:  the number of levels
    - f:  Indicator function
    - dt: Time between datapoints
    - ts: The timesteps

    """

    X_original = jnp.copy(X)
    funcs = []
    time_funcs = []
    Phis = []

    for l in range(L):
        X_residual = jnp.copy(X)
        Y_residual = jnp.copy(Y)

        J = 2**l
        r = M

        window_width = X.shape[1] // J
        stride = window_width // 2

        if X.shape[1] < r:
            print("Not enough timesteps for desired resolution")

            break

        num_windows = (X_original.shape[1]-window_width) // stride 
        print(f"{window_width=}")

        for j in range(-1, num_windows - 1):
           
            if j == -1:
                idx_l = 0
                idx_u = window_width // 2 + 1
            elif j == num_windows - 2:
                idx_l = stride * j
                idx_u = stride * (j + 1)
            else:
                idx_l = j * stride

                idx_u = jnp.minimum(idx_l + window_width + 1, X_original.shape[1])

            t_segment = ts[idx_l:idx_u]
            t_local = t_segment - t_segment[0]

            X_bin = X[:, idx_l:idx_u]
            Y_bin = Y[:, idx_l:idx_u]
            bin_relative_nergy = jnp.linalg.norm(X_bin) / jnp.linalg.norm(X_original[:, idx_l:idx_u])

            print("realtive energy", bin_relative_nergy)

            if bin_relative_nergy <= 1e-2:
                print(f"To low energy in Layer {l+1}, Bin {j+1}. Skipping.")

                continue

            # Compute DMD for each time bin
            Phi, b, omega = DMD_safe(X_bin, Y_bin, r, dt, energy_threshold)

            if Phi is None or jnp.any(jnp.isnan(b)) or jnp.any(jnp.isnan(omega)):
                print(f"Numerical instability detected in Layer {l+1}, Bin {j+1}. Skipping.")

                continue

            # Convert the eigenvalues and find the low frequency modes

            window_duration = jnp.abs((ts[idx_u] - ts[idx_l]))  # Watning the minus 1, may not be correct

            freq = jnp.abs(jnp.imag(omega)) / (2 * jnp.pi)

            if j == -1 or j ==num_windows - 2:
                mask = freq <= 4 / (2 * window_duration)
            else:
                mask = freq <= 4 / (window_duration)

            # Extract low frequency modes to the mrDMD function
            Phi_low = Phi[:, mask]
            b_low = b[mask]
            omega_low = omega[mask]
            omega_low = jnp.minimum(jnp.real(omega_low), 0.0) + 1j * jnp.imag(omega_low)
            
            #Continue if no modes are found
            if Phi_low.shape[1]==0:
                continue

            print(f"Saving {Phi_low.shape[1]} mode(s) in (j={j+1}, l={l+1})")
            for i in range(0, Phi_low.shape[1]):
                Phis.append(Phi_low[:, i][:, None])

            bin_start = float(ts[idx_l])

            bin_stop = float(ts[idx_u - 1])

            def robust_reconstruction(t, start=bin_start, stop=bin_stop, P=Phi_low, b=b_low, o=omega_low, wf = window_function, j=j, num_windows=num_windows):
                if j == -1:
                    shift = (stop - start)
                    #print(f"Original start= {start}, Shifted start = {start- shift}, end = {stop}")
                    window = wf(start - shift, stop, t)
                elif j == num_windows -2:
                    shift = (stop - start)
                    #print(f"Original start= {start}, Shifted start = {start}, end = {stop+shift}")
                    window = wf(start, stop + shift, t)
                else:
                    window = wf(start, stop, t)
                    
                # 1. Calculate the raw DMD reconstruction relative to bin start 
                #    (Using (t - start) is critical for phase alignment)
    
                dynamics = window * jnp.real(P @ (b[:, None] * jnp.exp(o[:, None] * (t - start))))              

                # 2. Make mask to fit on time bin
                mask = (t >= start) & (t < stop)

                # 3. Apply the mask
                return jnp.where(mask, dynamics, 0.0)

            funcs.append(robust_reconstruction)

            time_funcs.append(
                lambda t, start=ts[idx_l], stop=ts[min(idx_u, len(ts) - 1)], b_low=b_low, omega_low=omega_low, f=f: f(
                    start, stop, t
                )
                * (b_low[:, None] * jnp.exp(omega_low[:, None] * (t - start)))
            )

            # Reconstruct X using only the low frequency modes for each time bin

            if j == -1:
                shift = (t_segment[-1] - t_segment[0])
                #print(f"Original start= {t_segment[0]}, Shifted start = {t_segment[0]- shift}, end = {t_segment[-1]}")
                window = window_function(t_segment[0]- shift, t_segment[-1], t_segment)
            elif j == num_windows -2:
                shift = (t_segment[-1] - t_segment[0])
                #print(f"Original start= {t_segment[0]}, Shifted start = {t_segment[0]}, end = {t_segment[-1]+shift}")
                window = window_function(t_segment[0], t_segment[-1] + shift, t_segment)
            else:
                window = window_function(t_segment[0], t_segment[-1], t_segment)
            
            #print(window.shape, Phi_low.shape, t_segment.shape, b_low.shape)
            #print(jnp.outer(omega_low, t_local).shape)
           

            X_low = Phi_low @ (jnp.exp((jnp.outer(omega_low, t_local))) * window * b_low[:, None])
            Y_low = Phi_low @ (jnp.exp(jnp.outer(omega_low, t_local + dt)) * window * b_low[:, None])
            X_residual = X_residual.at[:, idx_l:idx_u].add(-X_low)
            Y_residual = Y_residual.at[:, idx_l:idx_u].add(-Y_low)

        X = X_residual
        Y = Y_residual

    # Sum all the functions together

    return Phis, lambda t: sum(g(t) for g in funcs), time_funcs


if __name__ == "__main__":
    from mr_dmd.helper_functions import sum_of_sines, indicator

    t_steps = 65 * 4
    n_steps = 20
    r = 8
    t_max = 64

    # g = lambda x,t : 1*jnp.cos(x + t)
    seed = 5
    g = sum_of_sines(seed, 10)


    window = hann_window
    f = indicator

    x = jnp.linspace(0, 2 * jnp.pi, n_steps)
    x_precise = jnp.linspace(0, 2 * jnp.pi, 500)
    t = jnp.linspace(0, t_max, t_steps)

    raw = g(x[:, None], t[None, :])
    X = raw[:, :-1]
    X_prime = raw[:, 1:]

    # Run the DMD
    L = 5
    M = r
    dt = t[1] - t[0]

    Phis, fun, _ = OmrDMD(X, X_prime, M, L, f, dt, t, energy_threshold=0.999, window_function= window)
    # Phi_DMD = DMD(X, X_prime, r)

    t1 = 5
    t2 = 50

    f_1 = fun(t1)
    f_10 = fun(t2)
    g_1 = g(x_precise, t1)
    g_10 = g(x_precise, t2)

    # fig, ax = plt.subplots(len(Phis), 1)
    # for i, phi in enumerate(Phis):
    #     ax[i].plot(jnp.real(phi))
    # # plt.show()

    print("Plotting")

    plt.plot(x_precise, jnp.real(g_1), label=f"true t={t1}")
    plt.scatter(x, jnp.real(f_1), label=f"reconstruction t={t1}", marker="x")
    plt.plot(x_precise, jnp.real(g_10), label=f"true t={t2}")
    plt.scatter(x, jnp.real(f_10), label=f"reconstruction t={t2}", marker="x")
    plt.legend()
    plt.show()


# tsteps = 64
# ts = jnp.linspace(1,64, 64, dtype= int)
# L = 3
# X = jnp.zeros((L,tsteps))


# for l in range(L):
#         print(f"Layer: {l+1} of {L}")
#         J = 4**l
#                                  # Ensure that r > 0                                      # Size of each time bin
#         ts_idx = jnp.linspace(0, tsteps, J+1, dtype=int)        # Splitting indecies

#         for j in range(0,J+1):

#             # Compute time segments
#             if j == 0:
#                 t_segment = ts[ts_idx[j]:ts_idx[j+1]]
#                 X = X.at[l,t_segment].add(0.5)
#             elif j == J:
#                 t_segment = ts[ts_idx[j-1]:ts_idx[j]]
#                 X = X.at[l,t_segment].add(0.5)
#             else:
#                 t_segment = ts[ts_idx[j-1]:ts_idx[j+1]]
#                 X = X.at[l,t_segment].add(0.5)
#             print(t_segment)

# # Create subplots for each layer
# fig, axes = plt.subplots(L, 1, figsize=(12, 3*L))
# if L == 1:
#     axes = [axes]

# for l in range(L):
#     axes[l].plot(ts, X[l])
#     axes[l].set_title(f"Layer {l+1}")
#     axes[l].set_xlabel("Time step")
#     axes[l].set_ylabel("Value")

# plt.tight_layout()
# plt.show()
