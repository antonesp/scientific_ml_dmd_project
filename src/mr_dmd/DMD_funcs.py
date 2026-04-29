import scipy
from matplotlib import pyplot as plt
import jax.numpy as jnp
import numpy as np
from time import sleep

def DMD(X,Xprime, r, energy_threshold=0.999):
    U, Sigma, VT = jnp.linalg.svd(X,full_matrices=0) # Step 1
    # Choose r adaptively based on singular value energy (prevents nans from forming)
    total_energy = jnp.sum(Sigma**2)
    cumulative_energy = jnp.cumsum(Sigma**2) / total_energy
    r_adaptive = int(jnp.searchsorted(cumulative_energy, energy_threshold)) + 1
    r_adaptive = min(r, r_adaptive, len(Sigma))  # never exceed requested r

    Ur = U[:,:r_adaptive]
    Sigmar = jnp.diag(Sigma[:r_adaptive])
    VTr = VT[:r_adaptive,:]
    Atilde = jnp.linalg.solve(Sigmar.T,(Ur.T @ Xprime @ VTr.T
    ).T).T # Step 2
    Lambda, W = jnp.linalg.eig(Atilde) # Step 3
    #Lambda = jnp.diag(Lambda)
    # Step 4
    Phi = Xprime @ jnp.linalg.solve(Sigmar.T,VTr).T @ W
    alpha1 = Sigmar @ VTr[:,0]
    b = jnp.linalg.solve(W @ jnp.diag(Lambda),alpha1)

    return Phi, Lambda, b


def fbDMD(X,Y, r):

    U,Sigma,VT = jnp.linalg.svd(X, full_matrices = 0) # Step 1

    Ur = U[:,:r]
    Sigmar = jnp.diag(Sigma[:r])
    VTr = VT[:r,:]

    U_rstar = jnp.conj(Ur.T)
    X_tilde = U_rstar @ X
    Y_tilde = U_rstar @ Y


    U_X, Sigma_X, Vstar_X = jnp.linalg.svd(X_tilde, full_matrices=0) 
    U_Y, Sigma_Y, Vstar_Y = jnp.linalg.svd(Y_tilde, full_matrices=0) 

    V_X = jnp.conj(Vstar_X.T)
    V_Y = jnp.conj(Vstar_Y.T)

  
    Sigma_X_inv = jnp.diag(1 / Sigma_X)
    Sigma_Y_inv = jnp.diag(1 / Sigma_Y)

    K_f_tilde = jnp.conj(U_X.T) @ Y_tilde @ V_X @ Sigma_X_inv
    K_b_tilde = jnp.conj(U_Y.T) @ X_tilde @ V_Y @ Sigma_Y_inv

    S_f = Y_tilde @ V_X @ Sigma_X_inv
    S_b = X_tilde @ V_Y @ Sigma_Y_inv

   
    K_f = S_f @ K_f_tilde @ jnp.linalg.pinv(S_f)
    K_b = S_b @ K_b_tilde @ jnp.linalg.pinv(S_b)

    K_tilde = scipy.linalg.sqrtm(K_f @ jnp.linalg.inv(K_b))

    Lambda, W = jnp.linalg.eig(K_tilde)
    Lambda = jnp.diag(Lambda)
    Phi = Y @ jnp.linalg.solve(Sigmar.T,VTr).T @ W

    return Phi, Lambda


def mrDMD(X, Y, M, L, f, dt, ts):
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
    T = ts.shape[0]                                                 # Number of time steps
   
    funcs = []
    for l in range(L):
        print(f"Layer: {l+1} of {L}")
        J = 2**l
        r = jnp.maximum(1, M//J)                                    # Ensure that r > 0                                      # Size of each time bin
        ts_idx = jnp.linspace(0, X.shape[1], J+1, dtype=int)        # Splitting indecies

        if X.shape[1] < r:
            print("Not enough timesteps for desired resolution")
            break
        
        X_temps = [] 
        for j in range(J):

            # Compute time segments
            t_segment = ts[ts_idx[j]:ts_idx[j+1]]
            t_local = t_segment - t_segment[0]

            X_bin = X[:, ts_idx[j]:ts_idx[j+1]]
            Y_bin = Y[:, ts_idx[j]:ts_idx[j+1]]
            
            # Compute DMD for each time bin
            Phi, Lambda, b = DMD(X_bin, Y_bin, r)
            print(Lambda)

            if X_bin.shape[0] < 2:
                X_temps.append(X_bin)  # passthrough, shape (n_spatial, n_time)
                continue

            # Convert the eigenvalues and find the low frequency modes
            window_duration = (ts_idx[j+1] - ts_idx[j]) * dt

            omega = jnp.log(Lambda +0j)/dt


            freq = jnp.abs(jnp.imag(omega))/(2*jnp.pi)
            mask = freq <= 1/window_duration

            # Extract low frequency modes to the mrDMD function
            Phi_low = Phi[:, mask]
            b_low = b[mask]
            omega_low = omega[mask]

            funcs.append(
                lambda t, start=ts[ts_idx[j]], stop = ts[min(ts_idx[j+1], len(ts)-1)], Phi_low=Phi_low, b_low=b_low, omega_low=omega_low, f = f:
                    f(start, stop, t) *
                    (Phi_low @ (b_low * jnp.exp(omega_low * (t-start))))
            )

            funcs.append(
                lambda t, start=ts[ts_idx[j]], stop = ts[min(ts_idx[j+1], len(ts)-1)], Phi_low=Phi_low, b_low=b_low, omega_low=omega_low, f = f:
                    f(start, stop, t) *
                    (Phi_low @ (b_low * jnp.exp(omega_low * (t-start))))
            )
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
            high_mask = ~mask
            if jnp.any(high_mask):
                Phi_high = Phi[:, high_mask]
                b_high = b[high_mask][:, None]
                omega_high = omega[high_mask][:, None]
                exp_term = jnp.exp(omega_high* t_local)
                X_temp = Phi_high @ (b_high * exp_term)
            else:
                X_temp = jnp.zeros((X.shape[0], len(t_segment)))

            X_temps.append(X_temp)
            print(X_temp.shape)

            
        # Combine X_temp to make new X and Y   
        X_full = jnp.concatenate(X_temps, axis = 1)
        ts = ts[:X_full.shape[1]]
        Y = X_full[:, 1:]
        X = X_full[:, :-1]

    # Sum all the functions together
    return lambda t: sum(g(t) for g in funcs)


if __name__ == "__main__":

    from mr_dmd.helper_functions import sum_of_sines

    t_steps = 200
    n_steps = 20
    r = 6
    t_max = 20


    g = lambda x,t : 1*jnp.cos(x + t)
    seed = 5
    #g = sum_of_sines(seed, 10)

    def f(start, stop, t):
        # This uses JAX logic to return 0 or 1 without Python if/else
        return jnp.where((t >= start) & (t <= stop), 1.0, 0.0)

    x = jnp.linspace(0, 2*jnp.pi, n_steps)
    x_precise = jnp.linspace(0, 2*jnp.pi, 500)
    t = jnp.linspace(0, t_max, t_steps)
    
    
  
    raw = (g(x[:, None],t[None, :]))
    X = raw[:,:-1]
    X_prime = raw[:, 1:]

    # Run the DMD
    L = 3
    M = r
    dt = t[1] - t[0]
    fun = mrDMD(X,X_prime,M, L, f, dt, t)

    t1 = 1
    t2 = 17    

    f_1 = fun(t1)
    f_10 = fun(t2)
    g_1 = g(x_precise, t1)
    g_10 = g(x_precise, t2)



    #plt.imshow(raw)
    #plt.show()
    print("Plotting")
    print(f_1)
    print(f_10)
    
    
    plt.plot(x_precise, jnp.real(g_1), label = f"true t={t1}")
    plt.scatter(x, jnp.real(f_1), label = f"reconstruction t={t1}", marker="x")
    plt.plot(x_precise, jnp.real(g_10), label = f"true t={t2}")
    plt.scatter(x, jnp.real(f_10), label = f"reconstruction t={t2}", marker="x")
    plt.legend()
    plt.show()
