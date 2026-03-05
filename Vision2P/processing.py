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


class DataProcessor:
    
    def load_dataset(self, path: str) -> Dataset:
        """
        Loads dataset from cache if exists, otherwise reads raw files.
        """
        cache_path = os.path.join(path, "dataset_cache.npz")
        
        if os.path.exists(cache_path):
            print("Loading dataset from cache...")
            with np.load(cache_path) as loader:
                return Dataset(
                    data=loader['data'],
                    angles=loader['angles'],
                    spatial_shape=tuple(loader['spatial_shape']),
                    parameters=loader['parameters']
                )
        
        print("Cache not found. Reading raw files (this may take a while)...")
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

    def preprocess_with_mask(self, dataset: Dataset, k: float = 0.25):
        """
        Filters by intensity and normalizes. Updates the dataset in-place.
        """

        pixel_intensities = np.linalg.norm(dataset.data, axis=1)
        threshold = k * np.nanmax(pixel_intensities)
        
        dataset.mask = pixel_intensities >= threshold
        dataset.kept_indices = np.where(dataset.mask)[0]

        dataset.data = dataset.data[dataset.mask]

        norms = np.linalg.norm(dataset.data, axis=1, keepdims=True)
        dataset.data = np.divide(dataset.data, norms, out=np.zeros_like(dataset.data), where=norms != 0)
        
        print(f"Preprocessing complete. Masked {np.sum(~dataset.mask)} noisy pixels.")

    def prepare_data_for_constraint(self, dataset: Dataset) -> np.ndarray:
        """
        Rearranges the feature axis of the dataset for 180° symmetry NMF.
        Returns the rearranged matrix X.
        """
        n_features = dataset.data.shape[1]
        if n_features % 2 != 0:
            raise ValueError("Number of angles must be even for symmetry constraint.")
            
        half = n_features // 2

        part_a = dataset.data[:, :half]
        part_b = dataset.data[:, half:]
        
        return np.hstack([part_a, part_b])
