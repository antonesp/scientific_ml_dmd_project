from matplotlib import pyplot as plt
from mr_dmd.DMD_funcs import mrDMD
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":

    ## Load the data
    data_path = "data/sst.npy"
    mask_path = "data/sst_mask.npy"

    sst_flat = jnp.load(data_path)
    mask = jnp.load(mask_path)
    sst_flat = sst_flat[:, 1680:1921]

    print(sst_flat.shape)
    X = sst_flat[:,:-1]
    Y = sst_flat[:, 1:]

    L = 4
    M = 200
    ts = jnp.arange(sst_flat.shape[0])
    dt = 1

    def f(start, stop, t):
        # This uses JAX logic to return 0 or 1 without Python if/else
        return jnp.where((t >= start) & (t <= stop), 1.0, 0.0)

    Phis, fun = mrDMD(X, Y, M, L, f, dt, ts)


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
    
    fig, ax = plt.subplots(len(Phis), 1)
    for i , phi in enumerate(Phis):
        print(phi.shape)
        full_phi = jnp.full(mask.shape, jnp.nan)
        full_phi = full_phi.at[~mask].set(phi[:, 0])
        ax[i].imshow(jnp.real(full_phi.reshape(180, 360)), cmap = "jet_r")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    im = ax.imshow(jnp.real(full_field) - jnp.real(true_field))
    ax.set_title("Difference")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()