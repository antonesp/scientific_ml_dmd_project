import numpy as np
import scipy


def DMD(X,Xprime,r):
    U,Sigma,VT = np.linalg.svd(X,full_matrices=0) # Step 1
    Ur = U[:,:r]
    Sigmar = np.diag(Sigma[:r])
    VTr = VT[:r,:]
    Atilde = np.linalg.solve(Sigmar.T,(Ur.T @ Xprime @ VTr.T
    ).T).T # Step 2
    Lambda, W = np.linalg.eig(Atilde) # Step 3
    Lambda = np.diag(Lambda)
    # Step 4
    Phi = Xprime @ np.linalg.solve(Sigmar.T,VTr).T @ W
    alpha1 = Sigmar @ VTr[:,0]
    b = np.linalg.solve(W @ Lambda,alpha1)
    return Phi, Lambda, b

def fbDMD(X,Y, r):

    U,Sigma,VT = np.linalg.svd(X, full_matrices = 0) # Step 1

    Ur = U[:,:r]
    Sigmar = np.diag(Sigma[:r])
    VTr = VT[:r,:]

    U_rstar = np.conj(Ur.T)
    X_tilde = U_rstar @ X
    Y_tilde = U_rstar @ Y


    U_X, Sigma_X, Vstar_X = np.linalg.svd(X_tilde, full_matrices=0) 
    U_Y, Sigma_Y, Vstar_Y = np.linalg.svd(Y_tilde, full_matrices=0) 

    V_X = np.conj(Vstar_X.T)
    V_Y = np.conj(Vstar_Y.T)

  
    Sigma_X_inv = np.diag(1 / Sigma_X)
    Sigma_Y_inv = np.diag(1 / Sigma_Y)

    K_f_tilde = np.conj(U_X.T) @ Y_tilde @ V_X @ Sigma_X_inv
    K_b_tilde = np.conj(U_Y.T) @ X_tilde @ V_Y @ Sigma_Y_inv

    S_f = Y_tilde @ V_X @ Sigma_X_inv
    S_b = X_tilde @ V_Y @ Sigma_Y_inv

   
    K_f = S_f @ K_f_tilde @ np.linalg.pinv(S_f)
    K_b = S_b @ K_b_tilde @ np.linalg.pinv(S_b)

    K_tilde = scipy.linalg.sqrtm(K_f @ np.linalg.inv(K_b))

    Lambda, W = np.linalg.eig(K_tilde)
    Lambda = np.diag(Lambda)
    Phi = Y @ np.linalg.solve(Sigmar.T,VTr).T @ W

    return Phi, Lambda