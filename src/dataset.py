import os
import numpy as np
import nrrd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

"""
python  src/dataset.py parotid_PDDCA+deepmind
"""


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(scan: np.ndarray, clip_min: float = -200, clip_max: float = 300) -> np.ndarray:
    """
    Clip HU range to soft-tissue window then z-score normalize per volume.
    Computed after cropping so stats reflect the parotid ROI, not the whole head.
    """
    scan = np.clip(scan, clip_min, clip_max).astype(np.float32)
    mean = scan.mean()
    std  = scan.std()
    return (scan - mean) / (std + 1e-8)


# ---------------------------------------------------------------------------
# Augmentation (applied identically to scan and mask)
# ---------------------------------------------------------------------------

def augment(scan: np.ndarray, mask: np.ndarray):
    """
    Simple augmentations that are safe for medical images.
    scan:  (D, H, W)  float32
    mask:  (1, D, H, W)  float32  [left channel]

    All flips are applied to both scan and mask consistently.
    Intensity jitter is scan-only.
    """
    # --- random flips along each axis ---
    # axis 0 = D (superior/inferior) — flip with 50% chance
    if np.random.rand() < 0.5:
        scan = np.flip(scan, axis=0).copy()
        mask = np.flip(mask, axis=1).copy()  # mask axis 1 = D (channel is axis 0)

    # axis 1 = H (anterior/posterior) — flip with 50% chance
    if np.random.rand() < 0.5:
        scan = np.flip(scan, axis=1).copy()
        mask = np.flip(mask, axis=2).copy()

    # axis 2 = W (left/right) — flip WITH label swap
    # flipping L/R means left parotid becomes right and vice versa
    if np.random.rand() < 0.5:
        scan = np.flip(scan, axis=2).copy()
        mask = np.flip(mask, axis=3).copy()

    # --- Gaussian noise ---
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.05, scan.shape).astype(np.float32)
        scan = scan + noise

    # --- intensity scaling ---
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        scan = scan * scale

    return scan, mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PDDCADataset(Dataset):
    """
    Loads pre-cropped PDDCA volumes (272 x 208 x 48) and their parotid masks.

    Expected structure:
        root/
            0522c0001/
                img.nrrd
                structures/
                    parotid_l.nrrd
                    parotid_r.nrrd
            0522c0002/
                ...

    Returns:
        scan:  (1, D, H, W) float32 tensor  — normalized CT
        mask:  (1, D, H, W) float32 tensor  — [left_parotid]
        name:  patient folder name (for debugging)
    """

    def __init__(self, root: str, patient_ids: list, augment: bool = False):
        self.root      = root
        self.ids       = patient_ids
        self.do_augment = augment

        # verify all expected files exist upfront so failures are loud
        for pid in self.ids:
            scan_path = os.path.join(root, pid, 'img.nrrd')
            left_path = os.path.join(root, pid, 'structures', 'Parotid_L.nrrd')
            # right_path= os.path.join(root, pid, 'structures', 'Parotid_R.nrrd')
            for p in [scan_path, left_path]: #, right_path]:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing file: {p}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx): 
        # each time replaces randomly an instance rather than adding a new one. Stacks a lot over 100s of epochs so data is differnt
        pid = self.ids[idx]

        scan,  _ = nrrd.read(os.path.join(self.root, pid, 'img.nrrd'))
        left,  _ = nrrd.read(os.path.join(self.root, pid, 'structures', 'Parotid_L.nrrd'))
        # right, _ = nrrd.read(os.path.join(self.root, pid, 'structures', 'Parotid_R.nrrd'))

        # nrrd loads as (W, H, D) by convention — transpose to (D, H, W)
        scan  = scan.transpose(2, 0, 1).astype(np.float32)   # (D, H, W)
        left  = left.transpose(2, 0, 1).astype(np.float32)
        # right = right.transpose(2, 0, 1).astype(np.float32)

        # normalize scan
        scan = normalize(scan)

        # stack masks into 2-channel array: (2, D, H, W)
        mask = np.stack([left], axis=0)

        # augment (training only)
        if self.do_augment:
            scan, mask = augment(scan, mask)

        # add channel dim to scan: (1, D, H, W)
        scan = scan[np.newaxis]

        # verify shape
        assert scan.shape == (1, 48, 208, 272), \
            f"Unexpected scan shape {scan.shape} for patient {pid}"
        assert mask.shape == (1, 48, 208, 272), \
            f"Unexpected mask shape {mask.shape} for patient {pid}"

        return (
            torch.from_numpy(scan),
            torch.from_numpy(mask),
            pid,
        )


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def make_splits(root: str, val_size: int = 5, test_size: int = 5, seed: int = 42):
    """
    Discovers all patient folders and splits into train/val/test.
    Returns three PDDCADataset instances.
    """
    all_ids = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    print(f"Found {len(all_ids)} patients: {all_ids}")

    assert len(all_ids) >= val_size + test_size + 1, \
        "Not enough patients for the requested split sizes."

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_ids).tolist()

    test_ids  = shuffled[:test_size]
    val_ids   = shuffled[test_size:test_size + val_size]
    train_ids = shuffled[test_size + val_size:]

    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    train_ds = PDDCADataset(root, train_ids, augment=True)
    val_ds   = PDDCADataset(root, val_ids,   augment=False)
    test_ds  = PDDCADataset(root, test_ids,  augment=False)

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "BSc-thesis/parotid_PDDCA+deepmind"

    train_ds, val_ds, test_ds = make_splits(root, val_size=5, test_size=5)

    # check one sample loads correctly
    scan, mask, pid = train_ds[0]
    print(f"\nPatient:    {pid}")
    print(f"Scan shape: {scan.shape}  dtype: {scan.dtype}")
    print(f"Mask shape: {mask.shape}  dtype: {mask.dtype}")
    print(f"Scan range: [{scan.min():.2f}, {scan.max():.2f}]")
    print(f"Left parotid voxels:  {mask[0].sum().int()}")
    # print(f"Right parotid voxels: {mask[1].sum().int()}")

    # check dataloader batching
    loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    scans, masks, pids = next(iter(loader))
    print(f"\nBatch scan shape: {scans.shape}")   # (2, 1, 48, 208, 272)
    print(f"Batch mask shape: {masks.shape}")   # (2, 2, 48, 208, 272)


    # ---------------------------------------------------------------------------
    # alignment sanity check
    # ---------------------------------------------------------------------------

    import matplotlib.pyplot as plt

    scan, mask, pid = train_ds[20]
    slice_idx = 24  # middle slice

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(scan[0, slice_idx], cmap='gray')
    axes[0].set_title('CT')
    axes[1].imshow(mask[0, slice_idx], cmap='hot')
    axes[1].set_title('Left parotid')
    # axes[2].imshow(mask[1, slice_idx], cmap='hot')
    # axes[2].set_title('Right parotid')
    plt.suptitle(pid)
    plt.show()