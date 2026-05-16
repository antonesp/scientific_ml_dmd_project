from matplotlib import pyplot as plt
from mr_dmd.DMD_funcs import mrDMD, DMD
import jax.numpy as jnp
import jax
import numpy as np
from mr_dmd.helper_functions import indicator

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    ## Load the data
    data_path = "data/sst.npy"
    mask_path = "data/sst_mask.npy"
    time_path = "data/time.npy"

    sst_flat = jnp.load(data_path)
    mask = jnp.load(mask_path)
    time = jnp.load(time_path)
    sst_flat = sst_flat[:, 1188:1428]
    time = time[1188:1428]
    t_steps = 1428 - 1188

    X = sst_flat[:, :-1]
    Y = sst_flat[:, 1:]

    L = 8
    M = 20
    r = 30
    ts = jnp.arange(sst_flat.shape[1])
    dt = 1

    # Mexican hat

    f = indicator
    Phis, fun, time_func = mrDMD(X, Y, M, L, f, dt, ts)
    r = len(Phis)
    Phis_DMD, Lambda, b = DMD(X, Y, r, energy_threshold=2)
    
    k = jnp.arange(t_steps)
    v_lambda = Lambda[:, None] ** k
    dynamics = b[:, None] * v_lambda

    X_rec = jnp.dot(Phis_DMD, dynamics)
    X_rec_real = jnp.real(X_rec)


    test_time = 190
    mrDMD_result = fun(test_time)

    full_field = jnp.full(mask.shape, jnp.nan)
    full_field = full_field.at[~mask].set(jnp.real(mrDMD_result.ravel()))  # plug DMD output back into valid locations
    full_field = full_field.reshape((180, 360))
    
    DMD_full_field = jnp.full(mask.shape, jnp.nan)
    DMD_full_field = DMD_full_field.at[~mask].set(X_rec_real[:, test_time])  # plug DMD output back into valid locations
    DMD_full_field = DMD_full_field.reshape((180, 360))

    true_field = jnp.full(mask.shape, jnp.nan)
    true_field = true_field.at[~mask].set(jnp.real(sst_flat[:, test_time]))
    true_field = true_field.reshape((180, 360))


    N1 = int(np.sqrt(len(Phis)))
    N2 = N1 + (len(Phis) - N1 * N1)

    fig, ax = plt.subplots(N1, N2)
    for i in range(N1):
        for j in range(N2):
            full_phi = jnp.full(mask.shape, jnp.nan)
            full_phi = full_phi.at[~mask].set(jnp.real(Phis[i * N1 + j]).ravel())
            im = ax[i, j].imshow(full_phi.reshape(180, 360), cmap="jet")
            fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(6, 5))

    im0 = ax[0,0].imshow(full_field)
    ax[0,0].set_title("mrDMD Reconstruction")
    ax[0,0].axis("off")
    im1 = ax[0,1].imshow(DMD_full_field)
    ax[0,1].set_title("DMD Reconstruction")
    ax[0,1].axis("off")
    im2 = ax[1,0].imshow(full_field - true_field)
    ax[1, 0].set_title("Difference mrDMD")
    ax[1, 0].axis("off")
    im3 = ax[1,1].imshow(DMD_full_field - true_field)
    ax[1,1].set_title("Difference DMD")
    ax[1,1].axis("off")
    fig.colorbar(im0, ax=ax[0,0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[1, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


    # Plot error
    mrDMD_reconstruct = jnp.zeros_like(sst_flat)
    for i in range(t_steps):
        X_at_step = fun(ts[i])[:X.shape[0]].ravel()
        mrDMD_reconstruct = mrDMD_reconstruct.at[:, i].set(X_at_step)


   
    error_DMD = jnp.mean((X_rec_real - sst_flat) ** 2, axis=0)
    error_mrDMD = jnp.mean((mrDMD_reconstruct - sst_flat) ** 2, axis=0)

    plt.plot(ts[:-2], error_DMD[:-2], label="Normal DMD")
    plt.plot(ts[:-2], error_mrDMD[:-2], label="mrDMD")
    plt.legend()
    plt.show()
