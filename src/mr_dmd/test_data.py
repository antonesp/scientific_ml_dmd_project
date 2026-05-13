from matplotlib import pyplot as plt
from mr_dmd.DMD_funcs import mrDMD
import jax.numpy as jnp
import jax
import numpy as np
from src.mr_dmd.helper_functions import indicator, make_wavelet_window

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":

    ## Load the data
    data_path = "data/sst.npy"
    mask_path = "data/sst_mask.npy"

    sst_flat = jnp.load(data_path)
    mask = jnp.load(mask_path)
    sst_flat = sst_flat[:, 1680:1921]
    

    X = sst_flat[:,:-1]
    Y = sst_flat[:, 1:]

    L = 4
    M = 7
    ts = jnp.arange(sst_flat.shape[0])
    dt = 1



    # Mexican hat
    f = make_wavelet_window('mexh')
    Phis, fun, time_func = mrDMD(X, Y, M, L, f, dt, ts)


    test_time = 120

    mrDMD_result = fun(120)

    print(mask.shape) 
    print((~mask).sum())
    print(mrDMD_result.shape)


    full_field = jnp.full(mask.shape, jnp.nan)
    full_field = full_field.at[~mask].set(mrDMD_result)  # plug DMD output back into valid locations
    

    full_field = full_field.reshape((180,360))

    true_field = jnp.full(mask.shape, jnp.nan)
    true_field = true_field.at[~mask].set(sst_flat[:,test_time])
    true_field = true_field.reshape((180,360))
    
    N1 = int(np.sqrt(len(Phis)))
    N2 = N1 + (len(Phis)-N1*N1)
    fig, ax = plt.subplots(N1, N2)
    for i in range(N1):
        for j in range(N2):
            full_phi = jnp.full(mask.shape, jnp.nan)
            full_phi = full_phi.at[~mask].set(Phis[i*N1 + j].ravel())
            im = ax[i, j].imshow(jnp.real(full_phi.reshape(180, 360)), cmap = "jet")
            fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(6, 5))

    im0 = ax[0].imshow(jnp.real(full_field))
    ax[0].set_title("Reconstruction")
    ax[0].axis("off")
    im1 = ax[1].imshow(jnp.real(true_field))
    ax[1].set_title("True")
    ax[1].axis("off")
    im2 = ax[2].imshow(jnp.real(full_field) - jnp.real(true_field))
    ax[2].set_title("Difference")
    ax[2].axis("off")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()