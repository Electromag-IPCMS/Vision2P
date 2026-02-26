import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import ticker

from dataclasses import dataclass

@dataclass
class DisplayConfig:
    color_map: str = "gnuplot2"
    savedir: str | None = None
    map_percent: bool = True
    pxsize: float = 0.1


@dataclass
class DecompositionResult:
    data: np.ndarray
    contributions: np.ndarray   # (n_samples, r)
    components: np.ndarray      # (r, n_features)

    @property
    def reconstruction(self):
        return self.contributions @ self.components


class ResultDisplay:

    def __init__(self, config: DisplayConfig):
        self.config = config

    def compare_reconstruction(self, result, image_shape=None):
        """
        Compare decomposition reconstruction with original data.

        Parameters
        ----------
        result : DecompositionResult
            Object containing the data, contributions, and components.
        image_shape : tuple (ny, nx), optional
            Shape to reshape the image for display. If None, assumes square.
        """
        
        data = result.data
        reconstructed = result.reconstruction

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

        for i in range(4):
            im = axes[i].imshow(maps[i], cmap=self.config.color_map,
                                vmin=(0 if i >= 2 else None))
            axes[i].set_title(titles[i], fontsize=14)
            axes[i].axis("off")

            if i >= 2:
                cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                cbar.locator = ticker.MaxNLocator(nbins=5)
                cbar.update_ticks()

        plt.tight_layout()

        if self.config.savedir is None:
            savepath = os.path.join(os.getcwd(), "nmf_reconstruction_error.png")
        else:
            os.makedirs(self.config.savedir, exist_ok=True)
            savepath = os.path.join(self.config.savedir, "nmf_reconstruction_error.png")

        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved to: {savepath}")