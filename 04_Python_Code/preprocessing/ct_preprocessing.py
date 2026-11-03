"""
CT Preprocessing Module:
Hounsfield Unit normalization, 3D volume resampling, lung parenchyma segmentation,
and 3D candidate patch extraction.
"""
import numpy as np
import scipy.ndimage

def normalize_hu(volume: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
    """
    Normalizes CT volume Hounsfield Units into [0.0, 1.0].
    I_norm = clip((I - HU_min) / (HU_max - HU_min), 0, 1)
    """
    volume = np.clip(volume, hu_min, hu_max)
    volume_norm = (volume - hu_min) / (hu_max - hu_min)
    return volume_norm.astype(np.float32)

def segment_lung_parenchyma(volume: np.ndarray, hu_threshold: float = -320.0) -> np.ndarray:
    """
    Segment lung tissue using density thresholding and morphological closing.
    """
    binary = volume < hu_threshold
    # Morphological cleaning
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    cleaned = scipy.ndimage.binary_opening(binary, structure=struct, iterations=2)
    cleaned = scipy.ndimage.binary_closing(cleaned, structure=struct, iterations=2)
    return cleaned.astype(np.uint8)

def resample_volume(volume: np.ndarray, current_spacing: tuple, target_spacing: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """
    Resample 3D volume to isotropic voxel spacing using cubic spline interpolation.
    """
    resize_factor = np.array(current_spacing) / np.array(target_spacing)
    new_shape = np.round(volume.shape * resize_factor).astype(int)
    real_resize_factor = new_shape / volume.shape
    resampled = scipy.ndimage.zoom(volume, real_resize_factor, order=3, mode='nearest')
    return resampled

def extract_3d_patch(volume: np.ndarray, center: tuple, patch_size: tuple = (32, 32, 32)) -> np.ndarray:
    """
    Extracts a 3D cubic patch centered at (z, y, x) with zero-padding if out of bounds.
    """
    cz, cy, cx = int(round(center[0])), int(round(center[1])), int(round(center[2]))
    pz, py, px = patch_size
    half_z, half_y, half_x = pz // 2, py // 2, px // 2

    z_min, z_max = cz - half_z, cz + half_z
    y_min, y_max = cy - half_y, cy + half_y
    x_min, x_max = cx - half_x, cx + half_x

    # Compute valid bounds and pads
    pad_z_before = max(0, -z_min)
    pad_y_before = max(0, -y_min)
    pad_x_before = max(0, -x_min)

    pad_z_after = max(0, z_max - volume.shape[0])
    pad_y_after = max(0, y_max - volume.shape[1])
    pad_x_after = max(0, x_max - volume.shape[2])

    v_zmin = max(0, z_min)
    v_zmax = min(volume.shape[0], z_max)
    v_ymin = max(0, y_min)
    v_ymax = min(volume.shape[1], y_max)
    v_xmin = max(0, x_min)
    v_xmax = min(volume.shape[2], x_max)

    patch_sub = volume[v_zmin:v_zmax, v_ymin:v_ymax, v_xmin:v_xmax]
    patch = np.pad(patch_sub,
                   ((pad_z_before, pad_z_after),
                    (pad_y_before, pad_y_after),
                    (pad_x_before, pad_x_after)),
                   mode='constant', constant_values=0)
    return patch[:pz, :py, :px].astype(np.float32)

def augment_3d_patch(patch: np.ndarray, rng: np.random.RandomState = None) -> np.ndarray:
    """
    3D spatial data augmentations: random flips (x, y, z), 90-degree rotations, Gaussian noise.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # Random flips
    if rng.rand() > 0.5:
        patch = np.flip(patch, axis=0)
    if rng.rand() > 0.5:
        patch = np.flip(patch, axis=1)
    if rng.rand() > 0.5:
        patch = np.flip(patch, axis=2)

    # Random 90 deg rotation in axial plane
    k = rng.randint(0, 4)
    if k > 0:
        patch = np.rot90(patch, k=k, axes=(1, 2))

    # Additive Gaussian noise
    if rng.rand() > 0.5:
        noise = rng.normal(0, 0.02, patch.shape).astype(np.float32)
        patch = np.clip(patch + noise, 0.0, 1.0)

    return np.ascontiguousarray(patch, dtype=np.float32)
