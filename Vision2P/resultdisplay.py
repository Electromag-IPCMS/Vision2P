import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from dataclasses import dataclass
from typing import Optional, Tuple
from .processing import Dataset

@dataclass
class DisplayConfig:
    """Configuration for result visualization."""
    color_map: str = "gnuplot2"
    savedir: Optional[str] = None
    map_percent: bool = True
    pxsize: Optional[float] = None
    image_shape: Optional[Tuple[int, int]] = None

@dataclass
class DecompositionResult:
    """Container for NMF output and original data."""
    data: np.ndarray
    contributions: np.ndarray
    components: np.ndarray

    @property
    def reconstruction(self):
        return self.contributions @ self.components

class ResultDisplay:
    """Class to handle the visualization of decomposition results."""

    def __init__(self, config: DisplayConfig):
        self.config = config

    def _sync_with_dataset(self, dataset: Dataset):
        """
        Synchronizes display configuration with dataset metadata.
        
        Args:
            dataset (Dataset): The dataset containing spatial and parameter info.
        """
        if self.config.image_shape is None:
            self.config.image_shape = dataset.spatial_shape
        
        if self.config.pxsize is None and dataset.parameters is not None:
            self.config.pxsize = dataset.parameters[2, 1] * 1000

    def reconstruct_spatial_map(self, result: DecompositionResult, dataset: Dataset, fill_value=np.nan) -> np.ndarray:
        """
        Reconstructs 2D spatial maps from masked NMF contributions.

        Args:
            result (DecompositionResult): The NMF result container.
            dataset (Dataset): The source dataset providing the intensity mask.
            fill_value: Value used for masked pixels.

        Returns:
            np.ndarray: Reconstructed maps with shape (n_components, height, width).
        """
        h, w = dataset.spatial_shape
        n_pixels = h * w
        n_components = result.contributions.shape[1]
        
        W_full = np.full((n_pixels, n_components), fill_value)
        
        if dataset.mask is not None:
            W_full[dataset.mask] = result.contributions
        else:
            W_full = result.contributions

        return W_full.reshape((h, w, n_components)).transpose(2, 0, 1)

    def plot_decomposition(self, result: DecompositionResult, dataset: Dataset, rearrange_symmetry: bool = False):
        """
        Visualizes spatial contribution maps (W) and polar components (H).

        Args:
            result (DecompositionResult): Object containing W and H matrices.
            dataset (Dataset): Dataset object providing angles and metadata.
            rearrange_symmetry (bool): If True, restores angular order for 180° symmetry.
        """
        self._sync_with_dataset(dataset)
        
        H = result.components
        if rearrange_symmetry:
            half = H.shape[1] // 2
            H = np.hstack([H[:, :half], H[:, half:]])
            
        W_maps = self.reconstruct_spatial_map(result, dataset)
        n_components = H.shape[0]
        im_ratio = W_maps[0].shape[0]/W_maps[0].shape[1]

        fig = plt.figure(figsize=(n_components * 5, 7))
        plt.rcParams.update({'font.size': 10})

        for i in range(n_components):
            ax_map = plt.subplot(2, n_components, i + 1)
            extent = None
            if self.config.pxsize:
                h, w = self.config.image_shape
                extent = [0, w * self.config.pxsize, 0, h * self.config.pxsize]
            
            im = ax_map.imshow(W_maps[i], cmap=self.config.color_map, extent=extent)
            if extent:
                ax_map.set_xlabel("µm")
                ax_map.set_ylabel("µm")
            else:
                ax_map.axis('off')
            plt.colorbar(im, ax=ax_map, fraction=0.047*im_ratio, pad=0.04)

            ax_polar = plt.subplot(2, n_components, n_components + i + 1, projection='polar')
            ax_polar.plot(dataset.angles, H[i, :], '-o', lw=2, ms=4, color='black', zorder=3)
            
            max_val = np.max(H[i, :])
            ax_polar.set_ylim(0, max_val * 1.1 if max_val > 0 else 1)
            ax_polar.yaxis.set_major_locator(ticker.MaxNLocator(3))

        plt.tight_layout()
        
        save_dir = self.config.savedir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "nmf_analysis.png"), dpi=300)
        plt.show()

    def compare_reconstruction(self, result: DecompositionResult, dataset: Dataset):
        """
        Compares original input data with NMF reconstruction across spatial dimensions.

        Args:
            result (DecompositionResult): The NMF result container.
            dataset (Dataset): The source dataset for spatial shape and masking.
        """
        self._sync_with_dataset(dataset)
        
        data_full = result.data
        recon_full = result.reconstruction
        image_shape = self.config.image_shape

        input_sum = np.sum(data_full, axis=1)
        recon_sum = np.sum(recon_full, axis=1)

        input_map = np.full(image_shape[0] * image_shape[1], np.nan)
        recon_map = np.full(image_shape[0] * image_shape[1], np.nan)
        
        input_map[dataset.mask] = input_sum
        recon_map[dataset.mask] = recon_sum
        
        input_map = input_map.reshape(image_shape)
        recon_map = recon_map.reshape(image_shape)

        eps = 1e-12
        error_percent = np.abs((recon_map - input_map) / (input_map + eps)) * 100
        im_ratio = maps[0].shape[0]/maps[0].shape[1]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ["Original (Sum)", "Reconstructed (Sum)", "Error %"]
        maps = [input_map, recon_map, error_percent]

        for i, ax in enumerate(axes):
            im = ax.imshow(maps[i], cmap=self.config.color_map)
            ax.set_title(titles[i])
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.047*im_ratio, pad=0.04)

        plt.tight_layout()
        plt.show()

    def plot_clusters_with_polar(self, result: DecompositionResult, dataset: Dataset, 
                                    threshold_ratio: float = 0.0, rearrange_symmetry: bool = False):
            """
            Displays a dominant cluster map where each pixel is assigned the color 
            of its strongest NMF component, alongside polar plots of those components.

            Args:
                result (DecompositionResult): Object containing W and H matrices.
                dataset (Dataset): Dataset object providing angles and metadata.
                threshold_ratio (float): Fraction of max intensity below which pixels are masked.
                rearrange_symmetry (bool): If True, restores angular order for 180° symmetry.
            """
            self._sync_with_dataset(dataset)
            
            W = result.contributions
            H = result.components
            
            if rearrange_symmetry:
                half = H.shape[1] // 2
                H = np.hstack([H[:, :half], H[:, half:]])
                
            n_pixels, n_components = W.shape
            h, w = self.config.image_shape

            clust_colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'cyan']
            if n_components > len(clust_colors):
                clust_colors = plt.cm.get_cmap('tab10').colors
                
            cmap = colors.ListedColormap(clust_colors[:n_components])
            norm = colors.BoundaryNorm(np.arange(n_components + 1), cmap.N)

            pixel_strength = np.sum(dataset.data, axis=1)
            intensity_mask = pixel_strength > (threshold_ratio * np.nanmax(pixel_strength))
            dominant_component = np.argmax(W, axis=1)

            full_map = np.full(h * w, np.nan)
            active_indices = dataset.kept_indices[intensity_mask] if dataset.mask is not None else np.where(intensity_mask)[0]
            full_map[active_indices] = dominant_component[intensity_mask]
            
            dominant_masked = np.ma.masked_invalid(full_map).reshape(h, w)
            fig = plt.figure(figsize=(4 * (n_components + 1), 5))
            plt.rcParams.update({'font.size': 10})

            ax0 = plt.subplot(1, n_components + 1, 1)
            extent = None
            if self.config.pxsize:
                extent = [0, w * self.config.pxsize, 0, h * self.config.pxsize]
                
            im = ax0.imshow(dominant_masked, cmap=cmap, norm=norm, interpolation='nearest', extent=extent)
            ax0.set_title("Dominant Clusters", fontsize=12, fontweight='bold')
            if extent:
                ax0.set_xlabel("µm")
                ax0.set_ylabel("µm")
            else:
                ax0.axis("off")

            for i in range(n_components):
                ax = plt.subplot(1, n_components + 1, i + 2, projection='polar')
                ax.plot(dataset.angles, H[i, :], '-o', lw=2, ms=4, color=clust_colors[i])           

                ax.tick_params(axis='y', labelsize=8)
                
                max_val = np.max(H[i, :])
                ax.set_ylim(0, max_val * 1.1 if max_val > 0 else 1)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(3))

            plt.tight_layout()
            
            save_dir = self.config.savedir or os.getcwd()
            os.makedirs(save_dir, exist_ok=True)
            savepath = os.path.join(save_dir, "nmf_clusters_polar.png")
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
            print(f"Cluster plot saved to: {savepath}")
            plt.show()