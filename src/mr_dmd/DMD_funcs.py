import scipy
from matplotlib import pyplot as plt
import jax.numpy as jnp
import numpy as np

def DMD(X,Xprime,r):
    U, Sigma, VT = jnp.linalg.svd(X,full_matrices=0) # Step 1

    Ur = U[:,:r]
    Sigmar = jnp.diag(Sigma[:r])
    VTr = VT[:r,:]
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
    - X:  Datapoints at t_n
    - Y:  Datapoints at t_n+1
    - M:  number of modes used in the first level when computing DMD
    - L:  the number of levels
    - f:  Indicator function
    - dt: Time between datapoints
    - ts: The timesteps
    """
    T = ts.shape[0]                                        # Number of time steps
   
    funcs = []
    for l in range(L):
        print(f"Layer: {l+1} of {L}")
        J = 2**l
        r = jnp.maximum(1, M//J)                                    # Ensure that r > 0
        time_split_size = T/J                                       # Size of each time bin
        ts_idx = jnp.linspace(0, X.shape[0], J+1, dtype=int)        # Splitting indecies

        if X.shape[0] < r:
            break
        
        X_temps = [] 
        for j in range(J):

            # Compute time segments
            t_segment = ts[ts_idx[j]:ts_idx[j+1]]
            t_local = t_segment - t_segment[0]

            X_bin = X[t_segment]
            Y_bin = Y[t_segment]
            
            # Compute DMD for each time bin
            Phi, Lambda, b = DMD(X_bin, Y_bin, r)

            if X_bin.shape[0] < 2:
                X_temps.append(X_bin.T)  # passthrough, shape (n_spatial, n_time)
                continue

            # Convert the eigenvalues and find the low frequency modes
            omega = jnp.log(Lambda)/dt
            freq = jnp.abs(jnp.imag(omega))/(2*jnp.pi)
            mask = freq <= 1/time_split_size

            # Extract low frequency modes to the mrDMD function
            Phi_low = Phi[:, mask]
            b_low = b[mask]
            omega_low = omega[mask]

            funcs.append(
                lambda t, start=ts[ts_idx[j]], stop = ts[ts_idx[j+1]], Phi_low=Phi_low, b_low=b_low, omega_low=omega_low:
                    f(start, stop, t) *
                    (Phi_low @ (b_low * jnp.exp(omega_low * t)))
            )

            # Reconstruct X using only the high frequency modes for each time bin
            high_mask = ~mask
            if jnp.any(high_mask):
                Phi_high = Phi[:, high_mask]
                b_high = b[high_mask][:, None]
                omega_high = omega[high_mask][:, None]
                exp_term = jnp.exp(omega_high* t_local)
                X_temp = Phi_high @ (b_high * exp_term)
            else:
                X_temp = jnp.zeros((X.shape[1], len(t_segment)))

            X_temps.append(X_temp)
            print(X_temp.shape)

            
        # Combine X_temp to make new X and Y   
        X_full = jnp.concatenate(X_temps, axis = 1)
        Y = X_full[:, 1:].T
        X = X_full[:, :-1].T

    # Sum all the functions together
    return lambda t: sum(g(t) for g in funcs)


if __name__ == "__main__":

    t_steps = 200
    n_steps = 50
    r = 10


    g = lambda x,t : 1 / (jnp.cosh(t*x+ 3))  + jnp.cos(x+t) + jnp.exp(t*1j)
    def f(start, stop, t):
        if (t < start) or (t > stop): return 0
        else: return 1

    x = jnp.linspace(0, 2*jnp.pi, n_steps)
    t = jnp.array(range(t_steps))
    

    raw = (g(x[:, None],t[None, :]))

    X = raw[:,:-1].T
    X_prime = raw[:, 1:].T

    # Run the DMD

    x = mrDMD(X,X_prime,r, 4, f, 1, t)


    f_10 = x(10)



    plt.plot(x, jnp.abs(f_10), label = "reconstruction")
    plt.plot(x, jnp.abs(raw[:, 10].T), label = "true")
    plt.legend()
    plt.show()
