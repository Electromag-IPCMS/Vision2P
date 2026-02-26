import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import ticker


def compare_nmf_reconstruction(data, H, W, image_shape=None, savedir=None):
    """
    Compare NMF reconstruction HW with original data.

    Parameters
    ----------
    data : array (n_pixels, n_angles)
    H : array (n_pixels, r)
    W : array (r, n_angles)
    image_shape : tuple (ny, nx), optional
        If None, assumes square image.
    savedir : str or None
        If None, saves in current working directory.
    """

    reconstructed = H @ W

    input_sum = np.sum(data, axis=1)
    recon_sum = np.sum(reconstructed, axis=1)

    n_pixels = data.shape[0]
    if image_shape is None:
        side = int(np.sqrt(n_pixels))
        if side * side != n_pixels:
            raise ValueError("Image shape not square. Please provide image_shape.")
        image_shape = (side, side)

    input_map = input_sum.reshape(image_shape)
    recon_map = recon_sum.reshape(image_shape)

    eps = 1e-12
    denom = np.where(input_map == 0, eps, input_map)
    error_percent = np.abs((recon_map - input_map) / denom) * 100

    rmse_per_pixel = np.sqrt(np.mean((reconstructed - data) ** 2, axis=1))
    rmse_map = rmse_per_pixel.reshape(image_shape)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    maps = [recon_map, input_map, error_percent, rmse_map]
    titles = ["Reconstructed", "Input", "Error %", "RMSE"]
    cmaps = ["gnuplot2", "gnuplot2", "gnuplot2", "gnuplot2"]

    for i in range(4):
        im = axes[i].imshow(maps[i], cmap=cmaps[i],
                            vmin=(0 if i >= 2 else None))
        axes[i].set_title(titles[i], fontsize=14)
        axes[i].axis("off")

        if i >= 2:
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.locator = ticker.MaxNLocator(nbins=5)
            cbar.update_ticks()

    plt.tight_layout()

    if savedir is None:
        savepath = os.path.join(os.getcwd(), "nmf_reconstruction_error.png")
    else:
        os.makedirs(savedir, exist_ok=True)
        savepath = os.path.join(savedir, "nmf_reconstruction_error.png")

    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved to: {savepath}")