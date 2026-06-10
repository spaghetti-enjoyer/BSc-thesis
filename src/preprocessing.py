import numpy as np
import nrrd

import os
import shutil

import re
import matplotlib.pyplot as plt

BLUE = '#378ADD'
GREEN = '#1D9E75'
ORANGE = '#D85A30'

import os
import shutil

PDDCA_ROOT    = "BSc-Thesis-Datasets/parotid_PDDCA+deepmind"
DEEPMIND_ROOT = "tcia-ct-scan-dataset/nrrds/test/oncologist"

# patients already in PDDCA
pddca_ids = set(os.listdir(PDDCA_ROOT))

merged  = []
skipped = []

for patient_id in sorted(os.listdir(DEEPMIND_ROOT)):
    src_dir = os.path.join(DEEPMIND_ROOT, patient_id)
    if not os.path.isdir(src_dir):
        continue

    # skip duplicates
    if patient_id in pddca_ids:
        skipped.append(patient_id)
        continue

    dst_dir        = os.path.join(PDDCA_ROOT, patient_id)
    dst_struct_dir = os.path.join(dst_dir, "structures")
    os.makedirs(dst_struct_dir, exist_ok=True)

    # --- CT scan: CT_IMAGE.nrrd -> img.nrrd ---
    src_ct = os.path.join(src_dir, "CT_IMAGE.nrrd")
    if not os.path.exists(src_ct):
        print(f"  WARNING: no CT_IMAGE.nrrd for {patient_id}, skipping.")
        shutil.rmtree(dst_dir)
        continue
    shutil.copy2(src_ct, os.path.join(dst_dir, "img.nrrd"))

    # --- parotid masks ---
    # TCGA patients use hyphens:  Parotid-Lt.nrrd / Parotid-Rt.nrrd
    # 0522c patients use underscores: Parotid_Lt.nrrd / Parotid_Rt.nrrd
    seg_dir = os.path.join(src_dir, "segmentations")

    if patient_id.startswith("TCGA"):
        src_L = os.path.join(seg_dir, "Parotid-Lt.nrrd")
        src_R = os.path.join(seg_dir, "Parotid-Rt.nrrd")
    else:
        src_L = os.path.join(seg_dir, "Parotid-Lt.nrrd")
        src_R = os.path.join(seg_dir, "Parotid-Rt.nrrd")

    missing = []
    if not os.path.exists(src_L): missing.append("Parotid_L")
    if not os.path.exists(src_R): missing.append("Parotid_R")
    if missing:
        print(f"  WARNING: {patient_id} missing {missing}, skipping.")
        shutil.rmtree(dst_dir)
        continue

    # rename to PDDCA convention
    shutil.copy2(src_L, os.path.join(dst_struct_dir, "Parotid_L.nrrd"))
    shutil.copy2(src_R, os.path.join(dst_struct_dir, "Parotid_R.nrrd"))

    merged.append(patient_id)
    print(f"  Merged: {patient_id}")

print(f"\nMerged:  {len(merged)} new patients: {merged}")
print(f"Skipped: {len(skipped)} duplicates: {skipped}")
print(f"Total in {PDDCA_ROOT}: {len(os.listdir(PDDCA_ROOT))}")

ROOT = "BSc-Thesis-Datasets/parotid_PDDCA+deepmind"
names = ["BrainStem.nrrd", "Chiasm.nrrd", "Mandible.nrrd", "OpticNerve_L.nrrd", "OpticNerve_R.nrrd", "Submandibular_L.nrrd", "Submandibular_R.nrrd"]

# delete the irrelevant segmentations
for root, dirs, files in os.walk(ROOT, topdown=False):
    for name in files:
        if name in names:
            full_path = os.path.join(root, name)
            os.remove(full_path)


# find and crop each scan individually
PATIENT_PREFIX = "0522c"
SCAN_NAME = "img.nrrd"
SUBDIR = "structures"
LEFT = "Parotid_L.nrrd"
RIGHT = "Parotid_R.nrrd"

LENGTH = 272
WIDTH = 208
HEIGHT = 48

LEFT_OFFSET = 10
RIGHT_OFFSET = LEFT_OFFSET
FRONT_OFFSET = 100
BACK_OFFSET = FRONT_OFFSET
TOP_OFFSET = 5
BOTTOM_OFFSET = TOP_OFFSET

# def get_mask_dimentions(coords):
#     return {
#         'x': coords['x'][1] - coords['x'][0],
#         'y': coords['y'][1] - coords['y'][0],
#         'z': coords['z'][1] - coords['z'][0]
#     }

def find_mask_coordinates(mask_volume):
    x, y, z = np.where(mask_volume > 0)
    return {
        'x': (x.min(), x.max()),
        'y': (y.min(), y.max()),
        'z': (z.min(), z.max()),
    }

def crop_scan(scan, left_mask_dims, right_mask_dims, target_shape):
    """
    scan: numpy array (X, Y, Z)
    left/right_mask_dims: dict with 'x', 'y', 'z' keys, each a (min, max) tuple
    target_shape: (W, L, H) tuple
    """
    target_w, target_l, target_h = target_shape
    scan_w, scan_l, scan_h = scan.shape

    def fit_to_target(lo, hi, target, scan_max):
        size = hi - lo
        if size < target:
            diff = target - size
            lo -= diff // 2
            hi += diff // 2 + diff % 2
        elif size > target:
            diff = size - target
            lo += diff // 2 + diff % 2
            hi -= diff // 2
        # clamp with compensating shift to preserve size
        if lo < 0:
            hi = min(scan_max, hi - lo)
            lo = 0
        if hi > scan_max:
            lo = max(0, lo - (hi - scan_max))
            hi = scan_max
        return lo, hi

    # --- compute raw edges from mask bounds + offsets ---
    right_edge  = right_mask_dims['x'][0] - RIGHT_OFFSET
    left_edge   = left_mask_dims['x'][1]  + LEFT_OFFSET
    front_edge  = min(left_mask_dims['y'][0], right_mask_dims['y'][0]) - FRONT_OFFSET
    back_edge   = max(left_mask_dims['y'][1], right_mask_dims['y'][1]) + BACK_OFFSET
    top_edge    = min(left_mask_dims['z'][0], right_mask_dims['z'][0]) - TOP_OFFSET
    bottom_edge = max(left_mask_dims['z'][1], right_mask_dims['z'][1]) + BOTTOM_OFFSET

    # --- fit each axis to target size ---
    right_edge, left_edge   = fit_to_target(right_edge, left_edge,   target_w, scan_w)
    front_edge, back_edge   = fit_to_target(front_edge, back_edge,   target_l, scan_l)
    top_edge,   bottom_edge = fit_to_target(top_edge,   bottom_edge, target_h, scan_h)

    # --- sanity check mask is still fully inside crop ---
    assert right_edge <= right_mask_dims['x'][0] and \
           left_edge  >= left_mask_dims['x'][1], \
        f"X crop cuts into mask: crop [{right_edge}:{left_edge}], " \
        f"mask [{right_mask_dims['x'][0]}:{left_mask_dims['x'][1]}]"
    assert front_edge <= min(left_mask_dims['y'][0], right_mask_dims['y'][0]) and \
           back_edge  >= max(left_mask_dims['y'][1], right_mask_dims['y'][1]), \
        f"Y crop cuts into mask: crop [{front_edge}:{back_edge}], " \
        f"mask [{min(left_mask_dims['y'][0], right_mask_dims['y'][0])}:" \
        f"{max(left_mask_dims['y'][1], right_mask_dims['y'][1])}]"
    assert top_edge    <= min(left_mask_dims['z'][0], right_mask_dims['z'][0]) and \
           bottom_edge >= max(left_mask_dims['z'][1], right_mask_dims['z'][1]), \
        f"Z crop cuts into mask: crop [{top_edge}:{bottom_edge}], " \
        f"mask [{min(left_mask_dims['z'][0], right_mask_dims['z'][0])}:" \
        f"{max(left_mask_dims['z'][1], right_mask_dims['z'][1])}]"

    return scan[right_edge:left_edge, front_edge:back_edge, top_edge:bottom_edge]

def plot_volume_stats(volume_dimensions):
    if volume_dimensions:
        data = volume_dimensions
    # else:
    #     data = {330: (192, 261, 31), 159: (187, 253, 34), 708: (148, 238, 36), 161: (180, 271, 34), 195: (138, 240, 26), 132: (145, 239, 31), 479: (164, 242, 27), 441: (153, 258, 54), 13: (150, 247, 26), 226: (180, 261, 39), 14: (155, 235, 33), 77: (172, 245, 34), 70: (157, 241, 29), 845: (149, 255, 41), 79: (152, 235, 31), 1: (130, 232, 25), 659: (140, 244, 37), 661: (172, 253, 33), 251: (167, 270, 32), 9: (156, 246, 26), 455: (153, 260, 34), 667: (158, 262, 32), 669: (150, 250, 35), 857: (162, 254, 36), 746: (172, 257, 37), 598: (148, 237, 32), 190: (147, 257, 31), 555: (182, 246, 38), 17: (151, 258, 31), 878: (159, 255, 30), 427: (197, 260, 36), 81: (191, 247, 31), 248: (156, 246, 34), 806: (183, 279, 31), 2: (157, 244, 28), 457: (148, 255, 37), 57: (172, 244, 31), 433: (141, 231, 29), 253: (185, 256, 34), 3: (141, 244, 28), 149: (203, 273, 31), 727: (155, 256, 32), 147: (217, 276, 52), 328: (167, 267, 30), 125: (184, 245, 26), 329: (123, 226, 21), 576: (158, 250, 32), 788: (157, 262, 30)}

    cases = sorted(data.keys())
    xs = [data[c][0] for c in cases]
    ys = [data[c][1] for c in cases]
    zs = [data[c][2] for c in cases]

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.scatter(range(len(cases)), xs, label='x (width)',  color=BLUE, s=20, linewidths=0)
    ax.scatter(range(len(cases)), ys, label='y (length)', color=GREEN, s=20, linewidths=0)
    ax.scatter(range(len(cases)), zs, label='z (height)', color=ORANGE, s=20, linewidths=0)

    # p95_x = np.percentile(xs, 95)
    # p95_y = np.percentile(ys, 95)
    # p95_z = np.percentile(zs, 95)

    # ax.axhline(p95_x, color=BLUE, linewidth=1, linestyle='--', alpha=0.5, label=f'x p95 ({p95_x:.0f})')
    # ax.axhline(p95_y, color=GREEN, linewidth=1, linestyle='--', alpha=0.5, label=f'y p95 ({p95_y:.0f})')
    # ax.axhline(p95_z, color=ORANGE, linewidth=1, linestyle='--', alpha=0.5, label=f'z p95 ({p95_z:.0f})')

    ax.axhline(208, color=BLUE, linewidth=1, linestyle='--', alpha=0.5, label=f'x ({272})')
    ax.axhline(272, color=GREEN, linewidth=1, linestyle='--', alpha=0.5, label=f'y ({208})')
    ax.axhline(48, color=ORANGE, linewidth=1, linestyle='--', alpha=0.5, label=f'z ({48})')

    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(cases, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('case id', fontsize=10)
    ax.set_ylabel('voxels', fontsize=10)

    # ax.legend(frameon=False, fontsize=9)
    ax.legend(frameon=False, fontsize=9, bbox_to_anchor=(1, 1), loc='upper left')
    # ax.legend(frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.25), loc='upper center', ncols=3)

    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('crop_dimensions.png', dpi=150)
    plt.show()



volume_dimensions = {}
patients = []

for root, dirs, files in os.walk(ROOT):
    for patient_directory in dirs:
        scan_path  = os.path.join(root, patient_directory, SCAN_NAME)
        mask_L_path = os.path.join(root, patient_directory, SUBDIR, LEFT)
        mask_R_path = os.path.join(root, patient_directory, SUBDIR, RIGHT)

        # skip if any required file is missing
        if not all(os.path.exists(p) for p in [scan_path, mask_L_path, mask_R_path]):
            continue

        print(scan_path, mask_L_path)

        # read raw scans
        scan_raw, header = nrrd.read(scan_path)
        mask_R_raw, _ = nrrd.read(mask_R_path)
        mask_L_raw, _ = nrrd.read(mask_L_path)

        # find dimensions of the masks
        mask_R_dims = find_mask_coordinates(mask_R_raw)
        mask_L_dims = find_mask_coordinates(mask_L_raw)

        # crop the scans
        target_shape = (WIDTH, LENGTH, HEIGHT)
        scan_raw = crop_scan(scan_raw, mask_L_dims, mask_R_dims, target_shape)
        mask_R_raw = crop_scan(mask_R_raw, mask_L_dims, mask_R_dims, target_shape)
        mask_L_raw = crop_scan(mask_L_raw, mask_L_dims, mask_R_dims, target_shape)

        # write them back to the files
        nrrd.write(scan_path, scan_raw, header)
        nrrd.write(mask_R_path, mask_R_raw)
        nrrd.write(mask_L_path, mask_L_raw)
        # nrrd.write("test.nrrd", scan_raw)

        # statistics
        if patient_directory.startswith("0522c"):
            patient_id = int(patient_directory[5:])
        else:
            # TCGA patients get a negative index based on sort order
            tcga_patients = sorted([p for p in os.listdir(ROOT) if p.startswith("TCGA")])
            patient_id = -(tcga_patients.index(patient_directory) + 1)

        volume_dimensions[patient_id] = scan_raw.shape

# print(sorted(patients), f"total number of patients: {len(patients)}", sep="\n")
plot_volume_stats(volume_dimensions)

