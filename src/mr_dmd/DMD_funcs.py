import scipy
from matplotlib import pyplot as plt
import jax.numpy as jnp



def DMD(X,Xprime, r, energy_threshold=0.999):
    U, Sigma, VT = jnp.linalg.svd(X,full_matrices=False) # Step 1
    # Choose r adaptively based on singular value energy (prevents nans from forming)
    total_energy = jnp.sum(Sigma)
    cumulative_energy = jnp.cumsum(Sigma) / total_energy
    r_adaptive = int(jnp.searchsorted(cumulative_energy, energy_threshold)) + 1
    r_adaptive = min(r, r_adaptive, len(Sigma))  # never exceed requested r

    Ur = U[:,:r_adaptive]
    Sigmar = jnp.diag(Sigma[:r_adaptive])

    Vr = VT[:r_adaptive,:].conj().T

    # Atilde = jnp.linalg.multi_dot([Ur.T, Xprime, Vr, jnp.linalg.inv(Sigmar)])
    Atilde = Ur.conj().T @ Xprime @ Vr @  jnp.linalg.inv(Sigmar)

    # Atilde = jnp.linalg.solve(Sigmar.T,(Ur.T @ Xprime @ VTr.T
    # ).T).T # Step 2

    Lambda, W = jnp.linalg.eig(Atilde) # Step 3
    
    # idx = Lambda.real.argsort()[::-1]
    #Lambda = jnp.diag(Lambda)
    # Step 4
    # Phi = Xprime @ jnp.linalg.solve(Sigmar.T,VTr).T @ W
    Phi = Xprime @ Vr @ jnp.linalg.inv(Sigmar) @ W
    # Phi = jnp.linalg.multi_dot([Xprime, Vr, jnp.linalg.inv(Sigmar), W])

    alpha1 = Sigmar @ Vr[0, :].conj().T

    b = jnp.linalg.lstsq(W @ jnp.diag(Lambda),alpha1, rcond= None)[0]

    return Phi, Lambda, b


def DMD_Allan(X,Xprime, r, energy_threshold=0.999):
    U, Sigma, VT = jnp.linalg.svd(X,full_matrices=False) # Step 1
    # Choose r adaptively based on singular value energy (prevents nans from forming)
    total_energy = jnp.sum(Sigma**2)
    cumulative_energy = jnp.cumsum(Sigma**2) / total_energy
    r_adaptive = int(jnp.searchsorted(cumulative_energy, energy_threshold)) + 1
    r_adaptive = min(r, r_adaptive, len(Sigma))  # never exceed requested r

    # Restrict the matrices to the found r-value
    Ur = U[:,:r_adaptive]
    Sigmar = jnp.diag(Sigma[:r_adaptive])
    Vr = VT[:r_adaptive,:].conj().T

    Atilde = Ur.conj().T @ Xprime @ Vr @  jnp.linalg.inv(Sigmar)

    Lambda, W = jnp.linalg.eig(Atilde) # Step 3
    Phi = Xprime @ Vr @ jnp.linalg.inv(Sigmar) @ W
    alpha1 = Sigmar @ Vr[0, :].conj().T

    b = jnp.linalg.lstsq(W @ jnp.diag(Lambda),alpha1, rcond= None)[0]

    return Phi, Lambda, b


def mrDMD(X, Y, M, L, f, dt, ts, energy_threshold=0.999):
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
   
    funcs = []
    time_funcs = []
    Phis = []
    for l in range(L):
        print(f"Layer: {l+1} of {L}")
        J = 2**l
        r = M                                    # Ensure that r > 0                                      # Size of each time bin
        ts_idx = jnp.linspace(0, X.shape[1], J+1, dtype=int)        # Splitting indecies

        if X.shape[1] < r:
            print("Not enough timesteps for desired resolution")
            break
        
        X_temps = [] 
        Y_temps = []
        for j in range(J):

            # Compute time segments
            t_segment = ts[ts_idx[j]:ts_idx[j+1]]
            t_local = t_segment - t_segment[0]

            X_bin = X[:, ts_idx[j]:ts_idx[j+1]]
            Y_bin = Y[:, ts_idx[j]:ts_idx[j+1]]
            
            # Compute DMD for each time bin
            Phi, Lambda, b = DMD(X_bin, Y_bin, r, energy_threshold)

            if X_bin.shape[0] < 2:
                X_temps.append(X_bin)  # passthrough, shape (n_spatial, n_time)
                Y_temps.append(Y_bin)
                continue

            # Convert the eigenvalues and find the low frequency modes
            window_duration = jnp.abs((ts[ts_idx[j+1]] - ts[ts_idx[j]]))/12
            omega = jnp.log(Lambda)/dt
            #print("Omega", omega)
            #print("Lambda", Lambda)
            freq = jnp.abs(jnp.imag(omega))/(2*jnp.pi)
            mask = freq <= 1/(2*window_duration)

            
            if mask.sum() == 0:         
                X_temps.append(X_bin)   # Continue if nothing is removed
                Y_temps.append(Y_bin)
                continue

            # Extract low frequency modes to the mrDMD function
            Phi_low = Phi[:, mask]
            b_low = b[mask]
            omega_low = omega[mask]
            print(f"Saving {Phi_low.shape[1]} modes in (j={j+1}, l={l+1})")
            for i in range(Phi_low.shape[1]):
                Phis.append(Phi_low[:, i][:, None])

            funcs.append(
                lambda t, start=ts[ts_idx[j]], stop = ts[min(ts_idx[j+1], len(ts)-1)], Phi_low=Phi_low, b_low=b_low, omega_low=omega_low, f = f:
                    f(start, stop, t) *
                    (Phi_low @ (b_low * jnp.exp(omega_low * (t-start))))
            )
            time_funcs.append(
                lambda t, start=ts[ts_idx[j]], stop = ts[min(ts_idx[j+1], len(ts)-1)], b_low=b_low, omega_low=omega_low, f = f:
                    f(start, stop, t) *
                    (b_low[:, None] * jnp.exp(omega_low[:, None] * (t - start))))
            """
            funcs.append(
                lambda t, start=ts[ts_idx[j]], stop=ts[min(ts_idx[j+1], len(ts)-1)], 
                    Phi_low=Phi_low, b_low=b_low, omega_low=omega_low:
                jnp.where(
                    (t >= start) & (t <= stop),
                    jnp.real(Phi_low @ (b_low * jnp.exp(omega_low * (t - start)))),
                    0.0
                )
            )
            """
            #print(funcs[-1](1))

            # Reconstruct X using only the high frequency modes for each time bin
          
            X_low = Phi_low @ (jnp.exp(jnp.outer(omega_low, t_segment))*b_low[:, None])
            Y_low = Phi_low @ (jnp.exp(jnp.outer(omega_low, (t_segment+dt)))*b_low[:, None])

            X_temp = X_bin - X_low
            Y_temp = Y_bin - Y_low

            X_temps.append(X_temp)
            Y_temps.append(Y_temp)

            
        # Combine X_temp to make new X and Y   
        X = jnp.concatenate(X_temps, axis = 1)
        Y = jnp.concatenate(Y_temps, axis = 1)
        ts = ts[:X.shape[1]]

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
    f = make_wavelet_window('gaus2')
    f = indicator
 

    x = jnp.linspace(0, 2*jnp.pi, n_steps)
    x_precise = jnp.linspace(0, 2*jnp.pi, 500)
    t = jnp.linspace(0, t_max, t_steps)
    
    
  
    raw = (g(x[:, None],t[None, :]))
    X = raw[:,:-1]
    X_prime = raw[:, 1:]

    # Run the DMD
    L = 4
    M = r
    dt = t[1] - t[0]
    
    Phis, fun, _ = mrDMD(X,X_prime,M, L, f, dt, t, energy_threshold= 0.999)
    #Phi_DMD = DMD(X, X_prime, r)

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

    #plt.imshow(raw)
    #plt.show()
    print("Plotting")
 
    
    
    plt.plot(x_precise, jnp.real(g_1), label = f"true t={t1}")
    plt.scatter(x, jnp.real(f_1), label = f"reconstruction t={t1}", marker="x")
    plt.plot(x_precise, jnp.real(g_10), label = f"true t={t2}")
    plt.scatter(x, jnp.real(f_10), label = f"reconstruction t={t2}", marker="x")
    plt.legend()
    plt.show()
