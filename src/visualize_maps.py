import napari
import nrrd
import numpy as np
import sys

patient = sys.argv[1] if len(sys.argv) > 1 else "0522c0416"
results_dir = f"results/{patient}"

# load all volumes
scan,     _ = nrrd.read(f"{results_dir}/scan.nrrd")
gt_l,     _ = nrrd.read(f"{results_dir}/gt_left.nrrd")
# gt_r,     _ = nrrd.read(f"{results_dir}/gt_right.nrrd")

basic_pred_l, _ = nrrd.read(f"{results_dir}/basic_pred_left.nrrd")
# basic_pred_r, _ = nrrd.read(f"{results_dir}/basic_pred_right.nrrd")
basic_sal_l,  _ = nrrd.read(f"{results_dir}/basic_saliency_left.nrrd")
# basic_sal_r,  _ = nrrd.read(f"{results_dir}/basic_saliency_right.nrrd")

# cbam_pred_l,  _ = nrrd.read(f"{results_dir}/cbam_pred_left.nrrd")
# cbam_pred_r,  _ = nrrd.read(f"{results_dir}/cbam_pred_right.nrrd")
# cbam_sal_l,   _ = nrrd.read(f"{results_dir}/cbam_saliency_left.nrrd")
# cbam_sal_r,   _ = nrrd.read(f"{results_dir}/cbam_saliency_right.nrrd")

# threshold predictions to binary
basic_mask_l = (basic_pred_l > 0.5).astype(np.uint8)
# basic_mask_r = (basic_pred_r > 0.5).astype(np.uint8)
# cbam_mask_l  = (cbam_pred_l  > 0.5).astype(np.uint8)
# cbam_mask_r  = (cbam_pred_r  > 0.5).astype(np.uint8)
# gt_both      = np.logical_or(gt_l, gt_r).astype(np.uint8)
gt_both      = gt_l.astype(np.uint8)

viewer = napari.Viewer(title=f"Patient {patient}")

# CT scan — base layer
viewer.add_image(scan, name="CT", colormap="gray", contrast_limits=[-1, 3])

# ground truth
viewer.add_labels(gt_both, name="GT both", opacity=0.4)

# # vanilla predictions
# viewer.add_labels(basic_mask_l, name="Basic pred L", opacity=0.4, color={1: "cyan"})
# viewer.add_labels(basic_mask_r, name="Basic pred R", opacity=0.4, color={1: "blue"})

# # CBAM predictions
# viewer.add_labels(cbam_mask_l, name="CBAM pred L", opacity=0.4, color={1: "orange"})
# viewer.add_labels(cbam_mask_r, name="CBAM pred R", opacity=0.4, color={1: "red"})

# replace add_labels with add_image for predictions
viewer.add_image(basic_mask_l.astype(np.float32), name="Basic pred L", 
                 colormap="cyan", opacity=0.4, contrast_limits=[0, 1])
# viewer.add_image(basic_mask_r.astype(np.float32), name="Basic pred R",
#                  colormap="blue", opacity=0.4, contrast_limits=[0, 1])
# viewer.add_image(cbam_mask_l.astype(np.float32),  name="CBAM pred L",
#                  colormap="green", opacity=0.4, contrast_limits=[0, 1])
# viewer.add_image(cbam_mask_r.astype(np.float32),  name="CBAM pred R",
#                  colormap="red", opacity=0.4, contrast_limits=[0, 1])

# saliency maps — vanilla
viewer.add_image(basic_sal_l, name="Basic saliency L", colormap="hot",
                 opacity=0.5, contrast_limits=[0, 1], visible=False)
# viewer.add_image(basic_sal_r, name="Basic saliency R", colormap="hot",
#                  opacity=0.5, contrast_limits=[0, 1], visible=False)

# saliency maps — CBAM
# viewer.add_image(cbam_sal_l, name="CBAM saliency L", colormap="magma",
#                  opacity=0.5, contrast_limits=[0, 1], visible=False)
# viewer.add_image(cbam_sal_r, name="CBAM saliency R", colormap="magma",
#                  opacity=0.5, contrast_limits=[0, 1], visible=False)

print(f"\nLayers loaded for {patient}:")
print("  Toggle saliency maps on/off in the layers panel on the left")
print("  Hot colormap = vanilla GBP")
print("  Magma colormap = CBAM GBP")

viewer.dims.order = (2, 1, 0)  # slice along H axis (left-right)
viewer.dims.current_step = (104, 0, 24)  # start at midline

napari.run()