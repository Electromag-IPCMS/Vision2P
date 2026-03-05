import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import colors
from matplotlib.colors import ListedColormap, BoundaryNorm

from dataclasses import dataclass
from typing import Optional

from processing import Dataset

@dataclass
class DisplayConfig:
    color_map: str = "gnuplot2"
    savedir: Optional[str] = None
    map_percent: bool = True
    pxsize: float = 0.1
    image_shape: Optional[tuple[int, int]] = None


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


    def compare_reconstruction(self, result: DecompositionResult):
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
        n_pixels = data.shape[0]
        image_shape = self.config.image_shape

        input_sum = np.sum(data, axis=1)
        recon_sum = np.sum(reconstructed, axis=1)

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
        print(f"Saved to: {savepath}")
        plt.show()


    def plot_decomposition(self, result: DecompositionResult, angles: np.ndarray, image_shape: Optional[tuple] = None):
        """
        Visualize NMF results: Spatial contribution maps (W) in the top row 
        and polar plots of components (H) in the bottom row.

        Parameters
        ----------
        result : DecompositionResult
            Object containing contributions (W) and components (H).
        angles : np.ndarray
            Array of angles in radians for the polar plots.
        image_shape : tuple (ny, nx), optional
            Shape to reshape the contribution vectors. If None, assumes square.
        """
        n_components = result.components.shape[0]
        n_pixels = result.contributions.shape[0]
        image_shape = self.config.image_shape

        if image_shape is None:
            side = int(np.sqrt(n_pixels))
            image_shape = (side, side)

        fig = plt.figure(figsize=(n_components * 5, 8))
        plt.rcParams.update({'font.size': 12})

        for i in range(n_components):
            ax_map = plt.subplot(2, n_components, i + 1)
            w_map = result.contributions[:, i].reshape(image_shape)
            im = ax_map.imshow(w_map, cmap=self.config.color_map)
            ax_map.axis('off')
            
            plt.colorbar(im, ax=ax_map, fraction=0.03, pad=0.04)

            ax_polar = plt.subplot(2, n_components, n_components + i + 1, projection='polar')
            ax_polar.plot(angles, result.components[i, :], '-o', lw=3, ms=5, zorder=1, color='black')
            data_h = result.components[i, :]
            max_val = np.max(data_h)
            if max_val > 0:
                upper_limit = np.ceil(max_val * 2) / 2 if max_val > 1 else np.ceil(max_val * 10) / 10
                if upper_limit == max_val: upper_limit += 0.1 
                
                ax_polar.set_ylim(0, upper_limit)
                ax_polar.yaxis.set_major_locator(ticker.LinearLocator(3))
            
            ax_polar.tick_params(axis='y', labelsize=10)

        plt.tight_layout()

        filename = "nmf_decomposition_analysis.png"
        save_dir = self.config.savedir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        savepath = os.path.join(save_dir, filename)

        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Decomposition plot saved to: {savepath}")
        plt.show()


    def plot_clusters_with_polar(self, result: DecompositionResult, angles: np.ndarray, threshold_ratio: float = 0.0):
            """
            Display dominant cluster map with polar plots of each component.
            Pixels below threshold are transparent.

            Parameters
            ----------
            result : DecompositionResult
                Object containing contributions (W) and components (H).
            angles : np.ndarray
                Array of angles in radians for the polar plots.
            image_shape : tuple (ny, nx), optional
                Shape to reshape the contribution vectors. If None, assumes square.
            threshold_ratio : float
                Fraction of max total intensity below which pixels are masked
            """
            W = result.contributions
            H = result.components
            n_pixels, n_components = W.shape
            
            image_shape = self.config.image_shape
            if image_shape is None:
                side = int(np.sqrt(n_pixels))
                if side * side != n_pixels:
                    raise ValueError("Image shape not square. Please provide image_shape in config.")
                image_shape = (side, side)
            
            clust_colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'cyan']
            if n_components > len(clust_colors):
                clust_colors = plt.cm.get_cmap('tab10').colors
                
            cmap = colors.ListedColormap(clust_colors[:n_components])
            norm = colors.BoundaryNorm(np.arange(n_components + 1), cmap.N)

            pixel_strength = np.sum(W, axis=1)
            threshold = threshold_ratio * pixel_strength.max()
            mask = pixel_strength > threshold
            
            dominant_component = np.argmax(W, axis=1)
            dominant_masked = np.ma.masked_where(~mask, dominant_component)

            fig = plt.figure(figsize=(4 * (n_components + 1), 5))

            ax0 = plt.subplot(1, n_components + 1, 1)
            im = ax0.imshow(dominant_masked.reshape(image_shape), cmap=cmap, norm=norm, interpolation='nearest')
            ax0.set_title("Dominant Clusters", fontsize=14, fontweight='bold')
            ax0.axis("off")


            for i in range(n_components):
                ax = plt.subplot(1, n_components + 1, i + 2, projection='polar')
                ax.plot(angles, H[i, :], '-o', lw=2, ms=4, color=clust_colors[i])
                ax.fill(angles, H[i, :], color=clust_colors[i], alpha=0.2)
                
                ax.set_title(f"Component {i}", color=clust_colors[i], fontsize=12, pad=10)
                ax.tick_params(axis='y', labelsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            filename = "nmf_clusters_polar.png"
            save_dir = self.config.savedir or os.getcwd()
            os.makedirs(save_dir, exist_ok=True)
            savepath = os.path.join(save_dir, filename)

            plt.savefig(savepath, dpi=300, bbox_inches="tight")
            print(f"Cluster plot saved to: {savepath}")
            plt.show()


    def rearrange_H(self, result: DecompositionResult) -> np.ndarray:
            """
            Restore original angular order to H matrix (for polar plots).
            Works symmetrically with prepare_data_for_constraint.
            """
            half = result.component.shape[1] // 2
            part_a = result.component[:, :half]
            part_b = result.component[:, half:]
            result.component = np.hstack([part_a, part_b])


    def reconstruct_spatial_map(self, result: DecompositionResult, dataset: Dataset, fill_value=np.nan):
            """
            Reconstructs 2D maps from masked NMF components.
            Returns: np.ndarray (n_components, height, width)
            """
            h, w = dataset.spatial_shape
            n_pixels = h * w
            n_components = result.contributions.shape[1]
            W_full = np.full((n_pixels, n_components), fill_value)

            W_full[dataset.mask] = result.contributions

            return W_full.reshape((h, w, n_components)).transpose(2, 0, 1)