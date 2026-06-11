import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv3d -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=False), # changed to not be in place so that the ydont conflict with gbp's hooks
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool3d then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsample then DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            # trilinear for 3D
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # pad x1 to match x2 size in case of odd dimensions
        # input is (B, C, D, H, W)
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, [
            diffW // 2, diffW - diffW // 2,
            diffH // 2, diffH - diffH // 2,
            diffD // 2, diffD - diffD // 2,
        ])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.

    Args:
        n_channels:  number of input channels (1 for single-modality CT)
        n_classes:   number of output segmentation classes (2 for left/right parotid)
        input_shape: (D, H, W) tuple — used only for the dimension check at init.
                     Each dimension must be divisible by 16 (4 pooling steps).
        bilinear:    use trilinear upsampling instead of transposed convolutions
        base_filters: number of filters in the first layer (default 64; use 32 to
                      save memory on smaller GPUs)
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        input_shape: tuple = (48, 196, 272),  # (D, H, W)
        bilinear: bool = False,
        base_filters: int = 64,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self._check_input_shape(input_shape)

        f = base_filters
        factor = 2 if bilinear else 1

        self.inc   = DoubleConv(n_channels, f)
        self.down1 = Down(f,         f * 2)
        self.down2 = Down(f * 2,     f * 4)
        self.down3 = Down(f * 4,     f * 8)
        self.down4 = Down(f * 8,     f * 16 // factor)

        self.up1   = Up(f * 16,      f * 8  // factor, bilinear)
        self.up2   = Up(f * 8,       f * 4  // factor, bilinear)
        self.up3   = Up(f * 4,       f * 2  // factor, bilinear)
        self.up4   = Up(f * 2,       f,                bilinear)

        self.outc  = OutConv(f, n_classes)

    @staticmethod
    def _check_input_shape(shape):
        for i, s in enumerate(shape):
            if s % 16 != 0:
                raise ValueError(
                    f"Input dimension {i} is {s}, which is not divisible by 16. "
                    f"With 4 downsampling steps each dimension must be a multiple of 16. "
                    f"Nearest valid values: {(s // 16) * 16} or {((s // 16) + 1) * 16}."
                )

    def forward(self, x):
        x1 = self.inc(x)      # (B, f,    D,    H,    W)
        x2 = self.down1(x1)   # (B, 2f,   D/2,  H/2,  W/2)
        x3 = self.down2(x2)   # (B, 4f,   D/4,  H/4,  W/4)
        x4 = self.down3(x3)   # (B, 8f,   D/8,  H/8,  W/8)
        x5 = self.down4(x4)   # (B, 16f,  D/16, H/16, W/16)

        x = self.up1(x5, x4)  # (B, 8f,   D/8,  H/8,  W/8)
        x = self.up2(x,  x3)  # (B, 4f,   D/4,  H/4,  W/4)
        x = self.up3(x,  x2)  # (B, 2f,   D/2,  H/2,  W/2)
        x = self.up4(x,  x1)  # (B, f,    D,    H,    W)

        logits = self.outc(x)          # (B, n_classes, D, H, W)
        return torch.sigmoid(logits)   # per-channel sigmoid for binary masks

    def use_checkpointing(self):
        """Gradient checkpointing to trade compute for memory."""
        self.inc   = torch.utils.checkpoint.checkpoint_wrapper(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint_wrapper(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint_wrapper(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint_wrapper(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint_wrapper(self.down4)
        self.up1   = torch.utils.checkpoint.checkpoint_wrapper(self.up1)
        self.up2   = torch.utils.checkpoint.checkpoint_wrapper(self.up2)
        self.up3   = torch.utils.checkpoint.checkpoint_wrapper(self.up3)
        self.up4   = torch.utils.checkpoint.checkpoint_wrapper(self.up4)
        self.outc  = torch.utils.checkpoint.checkpoint_wrapper(self.outc)


# ---------------------------------------------------------------------------
# Guided Backpropagation
# ---------------------------------------------------------------------------

class GuidedBackprop:
    """
    Guided backpropagation saliency maps for a trained UNet3D.

    How it works:
        During the backward pass, standard backprop allows both positive and
        negative gradients through ReLUs. GBP adds a second gate: gradients are
        only allowed through where the *forward activation* was also positive.
        This suppresses noise and produces sharper, boundary-focused maps.

    Usage:
        gbp = GuidedBackprop(model)
        saliency = gbp.generate(input_tensor, class_idx=0)  # 0=left, 1=right
        gbp.remove_hooks()  # always clean up after use
    """

    def __init__(self, model: UNet3D):
        self.model = model
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Walk every ReLU in the model and attach a backward hook that zeroes
        out gradients where either:
          - the incoming gradient is negative, OR
          - the forward activation was non-positive
        Both conditions must hold for a gradient to pass through (guided rule).
        """
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                # forward hook: stash the output (= post-ReLU activations)
                # so the backward hook can use them as a gate
                hook = module.register_forward_hook(self._forward_hook)
                self.hooks.append(hook)
                hook = module.register_full_backward_hook(self._backward_hook)
                self.hooks.append(hook)

    @staticmethod
    def _forward_hook(module, input, output):
        # store post-relu activations on the module for use in backward
        module._gbp_activation = output.detach()

    @staticmethod
    def _backward_hook(module, grad_input, grad_output):
        # gate 1: only positive incoming gradients (standard deconvnet rule)
        guided_grad = torch.clamp(grad_output[0], min=0.0)
        # gate 2: only where forward activation was positive (guided rule)
        guided_grad = guided_grad * (module._gbp_activation > 0).float()
        return (guided_grad,)

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int = 0,
        target_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute a GBP saliency map.

        Args:
            input_tensor: (1, 1, D, H, W) — single normalised CT volume,
                          requires_grad will be set automatically.
            class_idx:    which output channel to explain (0=left, 1=right parotid).
            target_mask:  optional (1, D, H, W) binary mask to restrict the
                          backward signal to predicted foreground voxels only.
                          If None, backprops from the sum of all predicted
                          foreground voxels for class_idx (recommended).

        Returns:
            saliency: (D, H, W) numpy array, values in [0, 1] after normalisation.
        """
        self.model.eval()

        x = input_tensor.clone()
        x.requires_grad_(True)

        # forward
        output = self.model(x)  # (1, 2, D, H, W), already sigmoid-ed

        # build scalar target: sum of predicted foreground for the chosen class
        pred_channel = output[0, class_idx]  # (D, H, W)
        if target_mask is not None:
            scalar = (pred_channel * target_mask.squeeze()).sum()
        else:
            # use predicted probability > 0.5 as the region of interest
            foreground = (pred_channel > 0.5).float()
            scalar = (pred_channel * foreground).sum()

        # backward
        self.model.zero_grad()
        scalar.backward()

        # the gradient w.r.t. the input is the saliency map
        saliency = x.grad[0, 0].detach().cpu()  # (D, H, W)

        # normalise to [0, 1] for visualisation
        saliency = torch.clamp(saliency, min=0.0)  # GBP only shows positives
        if saliency.max() > 0:
            saliency = saliency / saliency.max()

        return saliency.numpy()

    def remove_hooks(self):
        """Must be called after use to avoid memory leaks and hook accumulation."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        # clean up stored activations
        for module in self.model.modules():
            if isinstance(module, nn.ReLU) and hasattr(module, '_gbp_activation'):
                del module._gbp_activation


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    D, H, W = 48, 208, 272

    model = UNet3D(
        n_channels=1,
        n_classes=2,
        input_shape=(D, H, W),
        bilinear=False,
        base_filters=32,
    )

    dummy = torch.randn(1, 1, D, H, W)
    out = model(dummy)

    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- GBP sanity check ---
    gbp = GuidedBackprop(model)
    saliency = gbp.generate(dummy, class_idx=0)
    gbp.remove_hooks()

    print(f"Saliency map shape: {saliency.shape}")   # (48, 208, 272)
    print(f"Saliency range: [{saliency.min():.3f}, {saliency.max():.3f}]")