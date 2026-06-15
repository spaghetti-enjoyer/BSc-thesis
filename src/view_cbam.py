"""
view_cbam.py — napari viewer for CBAM attention maps produced by visualize_cbam.py

Usage:
    python src/view_cbam.py \
        --patient  0522c0416 \
        --output   results/ \
        --block    down4          # which CBAM block to inspect (default: down4 = bottleneck)

Controls:
    - Toggle layers on/off in the layers panel
    - Each top-k channel is a separate layer named by rank, channel index and weight
    - Spatial map is shown as a separate overlay
    - Use the slice slider to move through the volume
"""

import argparse
import os
import sys
import numpy as np
import nrrd
import napari


def load_nrrd(path):
    """Load nrrd and transpose from (W, H, D) back to (D, H, W)."""
    data, _ = nrrd.read(path)
    return data.transpose(2, 1, 0).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, required=True)
    parser.add_argument("--output",  type=str, default="results")
    parser.add_argument("--block",   type=str, default="down4",
                        choices=['inc', 'down1', 'down2', 'down3', 'down4',
                                 'up1', 'up2', 'up3', 'up4'])
    args = parser.parse_args()

    attn_dir  = os.path.join(args.output, args.patient, 'cbam_attention')
    block_dir = os.path.join(attn_dir, args.block)

    if not os.path.isdir(block_dir):
        print(f"No attention data found at {block_dir}")
        print(f"Available blocks: {sorted(os.listdir(attn_dir))}")
        sys.exit(1)

    # --- load scan ---
    scan = load_nrrd(os.path.join(attn_dir, 'scan.nrrd'))

    # --- load spatial map ---
    spatial_map = load_nrrd(os.path.join(block_dir, 'spatial_map.nrrd'))

    # --- load channel weights ---
    cw = np.load(os.path.join(block_dir, 'channel_weights.npy'))
    print(f"Channel weights — min: {cw.min():.3f}  max: {cw.max():.3f}  "
          f"mean: {cw.mean():.3f}")

    # --- load top-k attended maps ---
    top_files = sorted([
        f for f in os.listdir(block_dir)
        if f.startswith('top_') and f.endswith('.nrrd')
    ])
    print(f"Loading {len(top_files)} attended channel maps from block '{args.block}'")

    attended_maps = []
    for fname in top_files:
        data = load_nrrd(os.path.join(block_dir, fname))
        # parse rank, channel, weight from filename: top_00_ch042_w0.91.nrrd
        parts = fname.replace('.nrrd', '').split('_')
        rank   = int(parts[1])
        ch_idx = int(parts[2][2:])
        weight = float(parts[3][1:])
        attended_maps.append((rank, ch_idx, weight, data))

    # --- napari viewer ---
    viewer = napari.Viewer(title=f"CBAM attention — {args.patient} — {args.block}")

    # CT scan base
    viewer.add_image(scan, name="CT", colormap="gray", contrast_limits=[-1, 3])

    # spatial attention map
    viewer.add_image(
        spatial_map, name=f"spatial attention ({args.block})",
        colormap="magma", opacity=0.5, contrast_limits=[0, 1], visible=True
    )

    # top-k attended channel maps — all off by default, toggle to inspect
    colormaps = ['hot', 'cyan', 'green', 'bop orange', 'bop purple',
                 'blue', 'yellow', 'red', 'twilight', 'hsv']
    for rank, ch_idx, weight, data in attended_maps:
        cmap = colormaps[rank % len(colormaps)]
        viewer.add_image(
            data,
            name=f"ch{ch_idx:03d} w={weight:.2f} (rank {rank})",
            colormap=cmap,
            opacity=0.5,
            contrast_limits=[0, 1],
            visible=False,  # toggle on in layers panel
        )

    print(f"\nLayers loaded:")
    print(f"  CT scan")
    print(f"  Spatial attention map")
    for rank, ch_idx, weight, _ in attended_maps:
        print(f"  Rank {rank:2d} — channel {ch_idx:3d} — weight {weight:.3f}")

    print(f"\nTip: toggle channel layers on/off in the layers panel to compare")
    print(f"     Use --block to switch between encoder/decoder blocks")

    viewer.dims.order = (0, 1, 2)
    viewer.dims.current_step = (24, 0, 0)

    napari.run()