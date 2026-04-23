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


def mrDMD(X, Y, M, L, f, dt, T):
    """
    Multi resolution DMD function:
    - X:  Datapoints at t_n
    - Y:  Datapoints at t_n+1
    - M:  number of modes used in the first level when computing DMD
    - L:  the number of levels
    - f:  Indicator function
    - dt: Time between datapoints
    - T:  Total number of timesteps
    """
    
    x = lambda t: 0 * t

    
    for l in range(L):
        J = 2**l
        r = M/J
        time_split_size = T/J
        ts_idx = jnp.linspace(0, T, J+1)

        X_temp = lambda t: jnp.zeros_like(X)*t

        for j in range(J):
            Phi, Lambda, b = DMD(X[ts_idx[j]:ts_idx[j+1]], Y[ts_idx[j]:ts_idx[j+1]], r)
            omega = jnp.log(Lambda)/dt
            freq = jnp.abs(jnp.imag(omega))/(2*jnp.pi)
            mask = freq <= 1/time_split_size

            X_temp = Phi
        
            x = lambda t: x + f(l+1, j+1, t) * b[mask, j] * Phi[mask, j] * jnp.exp(omega[mask, j] * t)






    return 0


if __name__ == "__main__":

    t_steps = 200
    n_steps = 50
    r = 10


    f = lambda x,t : 1 / (jnp.cosh(t*x+ 3))  + jnp.cos(x+t) + jnp.exp(t*1j)

    x = jnp.linspace(0, 2*jnp.pi, n_steps)
    t = jnp.array(range(t_steps))

    raw = (f(x[:, None],t[None, :]))

    X = raw[:,:-1]
    X_prime = raw[:, 1:]

    # Run the DMD

    Phi, Lambda, b = DMD(X,X_prime,r)


    f_10 = Phi  @ jnp.linalg.matrix_power(Lambda, 10) @ b



    plt.plot(x, jnp.abs(f_10), label = "reconstruction")
    plt.plot(x, jnp.abs(raw[:, 10]), label = "true")
    plt.legend()
    plt.show()
