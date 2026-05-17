from matplotlib import pyplot as plt
from mr_dmd.DMD_funcs import mrDMD, DMD
import jax.numpy as jnp
import jax
import numpy as np
from mr_dmd.helper_functions import indicator
from matplotlib.widgets import Button

def browse_modes(Phis):
    """
    Phis: list of levels, each a list of bins, each a list of modes (n_spatial,) or (n_spatial, 1)
    """
    # Flatten to (level, bin, mode_idx, spatial_data)
    flat = []
    for l, level in enumerate(Phis):
        for j, bin_modes in enumerate(level):
            for m, phi in enumerate(bin_modes):
                phi_field = jnp.full(mask.shape, jnp.nan)
                phi_field = phi_field.at[~mask].set(jnp.real(phi.ravel()))  # plug DMD output back into valid locations
                phi_field = phi_field.reshape((180, 360))
                flat.append((l, j, m, phi_field))

    idx = [0]  # mutable index

    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    plt.subplots_adjust(bottom=0.2)

    def draw(i):
        l, j, m, phi = flat[i]
        
        axes.cla()
        axes.imshow(np.real(phi).reshape(shape), cmap='jet')
        axes.set_title(f"Real part — Level {l+1}, Bin {j+1}, Mode {m+1}")
        axes.axis("off")
       
        fig.suptitle(f"Mode {i+1} / {len(flat)}", fontsize=12)
        fig.canvas.draw()

    # You need to know the spatial shape to reshape — adjust this
    shape = (180, 360)  # e.g. for SST data

    ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
    ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')

    def next_mode(event):
        idx[0] = (idx[0] + 1) % len(flat)
        draw(idx[0])

    def prev_mode(event):
        idx[0] = (idx[0] - 1) % len(flat)
        draw(idx[0])

    btn_next.on_clicked(next_mode)
    btn_prev.on_clicked(prev_mode)

    draw(0)
    plt.show()



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
    t_steps = sst_flat.shape[1]


    X = sst_flat[:, :-1]
    Y = sst_flat[:, 1:]

    L = 8
    M = 20
    r = 30
    ts = jnp.arange(sst_flat.shape[1])
    dt = 1

    # Mexican hat

    f = indicator
    Phis, fun, time_func = mrDMD(X.copy(), Y.copy(), M, L, f, dt, ts)
    r = 0
    for phi_l in Phis:
        for phi_j in phi_l:
            r += 1

    Phis_DMD, Lambda, b = DMD(X, Y, r, energy_threshold=2)
    
    k = jnp.arange(t_steps)
    v_lambda = Lambda[:, None] ** k
    dynamics = b[:, None] * v_lambda

    X_rec = jnp.dot(Phis_DMD, dynamics)
    X_rec_real = jnp.real(X_rec)

    test_time = 120
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

    phi_1997 = Phis[3][2][2]    # El Nino
    phi_1999 = Phis[3][3][2]    # La Nina
    fig, ax = plt.subplots(1, 2)
    im1 = ax[0].imshow(phi_1997, cmap = "jet")
    im2 = ax[1].imshow(phi_1997, cmap = "jet")
    ax[0].set_title("El Niño")
    ax[1].set_title("La Niña")
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04, shrink = 0.7)
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04, shrink = 0.7)
    plt.tight_layout()
    plt.savefig("ENSO.png", bbox_inches="tight")
    plt.show()



    

    plt.figure()
    plt.imshow(true_field,  cmap='jet')
    plt.title("True sst")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04, shrink = 0.7)
    plt.savefig("true_sst.png", bbox_inches="tight")
    plt.show()

    # Differences
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im2 = ax[0].imshow(DMD_full_field - true_field,  cmap='jet')
    ax[0].set_title("Difference DMD")
    ax[0].axis("off")
    im3 = ax[1].imshow(full_field - true_field,  cmap='jet')
    ax[1].set_title("Difference mrDMD")
    ax[1].axis("off")

    fig.colorbar(im2, ax=ax[0], fraction=0.046, pad=0.04, shrink = 0.7)
    fig.colorbar(im3, ax=ax[1], fraction=0.046, pad=0.04, shrink = 0.7)
    plt.tight_layout()
    plt.savefig("sst_comparison.png", bbox_inches="tight")
    plt.show()

    # Plot error
    mrDMD_reconstruct = jnp.zeros_like(sst_flat)
    for i in range(t_steps):
        X_at_step = fun(ts[i])[:X.shape[0]].ravel()
        mrDMD_reconstruct = mrDMD_reconstruct.at[:, i].set(X_at_step)


   
    error_DMD = jnp.mean((X_rec_real - sst_flat) ** 2, axis=0)
    error_mrDMD = jnp.mean((mrDMD_reconstruct - sst_flat) ** 2, axis=0)
    max_error_idx = jnp.argmax(error_mrDMD[:-2])
    print(ts[max_error_idx])

    plt.plot(ts[:-2], error_DMD[:-2], label="Normal DMD")
    plt.plot(ts[:-2], error_mrDMD[:-2], label="mrDMD")
    plt.legend()
    plt.savefig("sst_error.png")
    plt.show()
