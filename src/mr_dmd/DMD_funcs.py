from matplotlib import pyplot as plt
import jax.numpy as jnp



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
        return None, None, None, None

    # 2. Perform standard DMD
    Phi, Lambda, b = DMD(X, Y, r, energy_threshold)

    # 3. Handle Numerical Singularities
    Lambda_clipped = jnp.where(jnp.abs(Lambda) < 1e-7, 1e-7, Lambda)  # Clip Lambda to prevent log(0)
    # Lambda_clipped = Lambda

    # 4. Convert to Omega and CLAMP
    omega = jnp.log(Lambda_clipped) / dt
    omega = jnp.minimum(jnp.real(omega), 0.0) + 1j * jnp.imag(
        omega
    )  # Force Re(omega) <= 0 to prevent the explosion at t=64

    return Phi, b, omega


def mrDMD(X, Y, M, L, f, dt, ts, energy_threshold=0.999, window = False):
    """
    Multi resolution DMD function:
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
        print(f"Layer: {l+1} of {L}")
        J = 2**l
        r = M  # Ensure that r > 0                                      # Size of each time bin
        ts_idx = jnp.linspace(0, X.shape[1], J + 1, dtype=int)  # Splitting indecies

        if X.shape[1] < r:
            print("Not enough timesteps for desired resolution")
            break

        X_temps = []
        Y_temps = []
        for j in range(J):
            # Compute time segments
            t_segment = ts[ts_idx[j] : ts_idx[j + 1]]
            t_local = t_segment - t_segment[0]

            X_bin = X[:, ts_idx[j] : ts_idx[j + 1]]
            Y_bin = Y[:, ts_idx[j] : ts_idx[j + 1]]

            bin_relative_nergy = jnp.linalg.norm(X_bin) / jnp.linalg.norm(X_original[:, ts_idx[j] : ts_idx[j + 1]])
            print("realtive energy", bin_relative_nergy)
            if bin_relative_nergy <= 1e-2:
                print(f"To low energy in Layer {l+1}, Bin {j+1}. Skipping.")
                X_temps.append(jnp.zeros_like(X_bin))
                Y_temps.append(jnp.zeros_like(Y_bin))
                continue

            # Compute DMD for each time bin
            Phi, b, omega = DMD_safe(X_bin, Y_bin, r, dt, energy_threshold)

            if Phi is None or jnp.any(jnp.isnan(b)) or jnp.any(jnp.isnan(omega)):
                print(f"Numerical instability detected in Layer {l+1}, Bin {j+1}. Skipping.")
                X_temps.append(X_bin)  # Pass the data forward as residual instead
                Y_temps.append(Y_bin)
                continue

            if Phi is None:
                print("Phi was nan")
                X_temps.append(X_bin)
                Y_temps.append(Y_bin)
                continue

            if X_bin.shape[0] < 2:
                X_temps.append(X_bin)  # passthrough, shape (n_spatial, n_time)
                Y_temps.append(Y_bin)
                continue

            # Convert the eigenvalues and find the low frequency modes
            window_duration = jnp.abs((ts[ts_idx[j + 1]] - ts[ts_idx[j]]))  # Watning the minus 1, may not be correct
            freq = jnp.abs(jnp.imag(omega)) / (2 * jnp.pi)
            mask = freq <= 1 / (window_duration)

            if mask.sum() == 0:
                X_temps.append(X_bin)  # Continue if nothing is removed
                Y_temps.append(Y_bin)
                continue

            # Extract low frequency modes to the mrDMD function
            Phi_low = Phi[:, mask]
            b_low = b[mask]
            omega_low = omega[mask]
            omega_low = jnp.minimum(jnp.real(omega_low), 0.0) + 1j * jnp.imag(omega_low)

            print(f"Saving {Phi_low.shape[1]} mode(s) in (j={j+1}, l={l+1})")
            for i in range(0, Phi_low.shape[1]):
                Phis.append(Phi_low[:, i][:, None])

            def robust_reconstruction(
                t, start=ts[ts_idx[j]], stop=ts[ts_idx[j + 1] - 1], P=Phi_low, b=b_low, o=omega_low
            ):
                # This prevents any NaN in P, b, or o from leaking into t < start or t > stop
                dynamics = jnp.real(P @ (b[:, None] * jnp.exp(o[:, None] * (t - start))))              

                # 2. Make mask to fit on time bin
                mask = (t >= start) & (t <= stop)

                # 3. Apply the mask
                return jnp.where(mask, dynamics, 0.0)
                

            funcs.append(robust_reconstruction)
            time_funcs.append(
                lambda t,
                start=ts[ts_idx[j]],
                stop=ts[min(ts_idx[j + 1], len(ts) - 1)],
                b_low=b_low,
                omega_low=omega_low,
                f=f: f(start, stop, t) * (b_low[:, None] * jnp.exp(omega_low[:, None] * (t - start)))
            )

            # Reconstruct X using only the low frequency modes for each time bin
            X_low = Phi_low @ (jnp.exp(jnp.outer(omega_low, t_local)) * b_low[:, None])

            X_temp = X_bin - X_low
            Y_temp = Y_bin - (Phi_low @ (jnp.exp(jnp.outer(omega_low, (t_local + dt))) * b_low[:, None]))

            X_temps.append(X_temp)
            Y_temps.append(Y_temp)

        # Combine X_temp to make new X and Y
        X = jnp.concatenate(X_temps, axis=1)
        Y = jnp.concatenate(Y_temps, axis=1)
        ts = ts[: X.shape[1]]

    # Sum all the functions together
    return Phis, lambda t: sum(g(t) for g in funcs), time_funcs


if __name__ == "__main__":
    from mr_dmd.helper_functions import sum_of_sines, make_wavelet_window, indicator

    t_steps = 200
    n_steps = 20
    r = 2
    t_max = 20

    # g = lambda x,t : 1*jnp.cos(x + t)
    seed = 5
    g = sum_of_sines(seed, 10)

    # Mexican hat
    f = make_wavelet_window("gaus2")
    f = indicator

    x = jnp.linspace(0, 2 * jnp.pi, n_steps)
    x_precise = jnp.linspace(0, 2 * jnp.pi, 500)
    t = jnp.linspace(0, t_max, t_steps)

    raw = g(x[:, None], t[None, :])
    X = raw[:, :-1]
    X_prime = raw[:, 1:]

    # Run the DMD
    L = 4
    M = r
    dt = t[1] - t[0]

    Phis, fun, _ = mrDMD(X, X_prime, M, L, f, dt, t, energy_threshold=0.999)
    # Phi_DMD = DMD(X, X_prime, r)

    t1 = 1
    t2 = 17

    f_1 = fun(t1)
    f_10 = fun(t2)
    g_1 = g(x_precise, t1)
    g_10 = g(x_precise, t2)

    fig, ax = plt.subplots(len(Phis), 1)
    for i, phi in enumerate(Phis):
        ax[i].plot(jnp.real(phi))
    plt.show()

    print("Plotting")

    plt.plot(x_precise, jnp.real(g_1), label=f"true t={t1}")
    plt.scatter(x, jnp.real(f_1), label=f"reconstruction t={t1}", marker="x")
    plt.plot(x_precise, jnp.real(g_10), label=f"true t={t2}")
    plt.scatter(x, jnp.real(f_10), label=f"reconstruction t={t2}", marker="x")
    plt.legend()
    plt.show()
