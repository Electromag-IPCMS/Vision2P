import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Dataset:
    data: np.ndarray           # (n_pixels, n_features)
    angles: np.ndarray         # (n_features,)
    spatial_shape: Tuple[int, int]
    parameters: np.ndarray
    mask: Optional[np.ndarray] = None
    kept_indices: Optional[np.ndarray] = None

    def save_cache(self, path: str):
        """Saves the essential raw data to a compressed numpy file."""
        cache_path = os.path.join(path, "dataset_cache.npz")
        np.savez_compressed(
            cache_path, 
            data=self.data, 
            angles=self.angles, 
            spatial_shape=self.spatial_shape,
            parameters=self.parameters
        )
        print(f"Cache created at: {cache_path}")

    def l2_normalize(self):
        """Applies L2 normalization to data in-place."""
        norms = np.linalg.norm(self.data, axis=1, keepdims=True)
        self.data = np.divide(self.data, norms, out=np.zeros_like(self.data), where=norms != 0)
        print("Data L2-normalized.")
        return self 

    def apply_threshold_mask(self, k: float = 0.25):
        """Calculates and applies intensity mask in-place."""
        pixel_intensities = np.linalg.norm(self.data, axis=1)
        threshold = k * np.nanmax(pixel_intensities)
        
        self.mask = pixel_intensities >= threshold
        self.kept_indices = np.where(self.mask)[0]

        self.data = self.data[self.mask]
        print(f"Mask applied: {self.data.shape[0]} pixels remaining.")
        return self

    def reconstruct_spatial_map(self, W_masked: np.ndarray, fill_value=np.nan) -> np.ndarray:
        """
        Reconstructs the 2D spatial maps using the internal mask.
        Returns: (n_components, height, width)
        """
        if self.mask is None:
            h, w = self.spatial_shape
            return W_masked.T.reshape(-1, h, w)

        h, w = self.spatial_shape
        n_pixels = h * w
        n_components = W_masked.shape[1]

        W_full = np.full((n_pixels, n_components), fill_value)
        W_full[self.mask] = W_masked

        return W_full.reshape((h, w, n_components)).transpose(2, 0, 1)

    def prepare_data_for_constraint(self) -> np.ndarray:
        """Rearranges the feature axis for 180° symmetry NMF."""
        n_features = self.data.shape[1]
        if n_features % 2 != 0:
            raise ValueError("Number of angles must be even for symmetry constraint.")
        
        half = n_features // 2
        return np.hstack([self.data[:, :half], self.data[:, half:]])
    

class DataProcessor:
    
    def load_dataset(self, path: str) -> Dataset:
        """
        Loads dataset from cache if exists, otherwise reads raw files.
        """
        cache_path = os.path.join(path, "dataset_cache.npz")
        
        if os.path.exists(cache_path):
            print("Loading dataset from cache.")
            with np.load(cache_path) as loader:
                return Dataset(
                    data=loader['data'],
                    angles=loader['angles'],
                    spatial_shape=tuple(loader['spatial_shape']),
                    parameters=loader['parameters']
                )
        
        print("Cache not found. Reading raw files (this may take a while).")
        dataset = self._read_raw_files(path)
        dataset.save_cache(path)
        return dataset


    def _read_raw_files(self, path: str) -> Dataset:
        """Internal method to parse DAT folder and parameters."""
        data_path = os.path.join(path, "DAT/")
        files = sorted([f for f in os.listdir(data_path) if f.endswith(('.dat', '.txt'))])
        
        raw_list = [np.loadtxt(os.path.join(data_path, f)) for f in files]
        raw_stack = np.stack(raw_list, axis=2) # (H, W, N_angles)
        
        h, w, n_angles = raw_stack.shape
        data_flat = raw_stack.reshape(-1, n_angles)
        
        parameters = np.loadtxt(os.path.join(path, "parameters.dat"))
        
        angles = np.radians(parameters[0, 4] + np.arange(num_angles := int(parameters[1, 4])) * parameters[2, 4])
        
        return Dataset(
            data=data_flat,
            angles=angles,
            spatial_shape=(h, w),
            parameters=parameters
        )


