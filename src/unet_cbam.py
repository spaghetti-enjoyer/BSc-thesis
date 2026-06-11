import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CBAM (3D)
# ---------------------------------------------------------------------------

class ChannelAttention3D(nn.Module):
    """
    Channel attention: learns which feature channels to emphasise.
    Uses both avg and max pooling across spatial dims, passed through
    a shared 2-layer MLP, then summed and sigmoid-gated.
    """
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        # shared MLP — implemented as 1x1x1 convs
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, max(in_planes // ratio, 1), kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv3d(max(in_planes // ratio, 1), in_planes, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)   # (B, C, 1, 1, 1)


class SpatialAttention3D(nn.Module):
    """
    Spatial attention: learns where in the volume to focus.
    Computes channel-wise avg and max, concatenates, then convolves
    to a single-channel spatial gate.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)          # (B, 1, D, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)        # (B, 1, D, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)           # (B, 2, D, H, W)
        return self.sigmoid(self.conv(x_cat))                  # (B, 1, D, H, W)


class CBAM3D(nn.Module):
    """
    Full CBAM block: channel attention then spatial attention.
    Applied as: x = x * channel_gate(x)
                x = x * spatial_gate(x)
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention3D(in_planes, ratio)
        self.spatial_att = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# ---------------------------------------------------------------------------
# U-Net parts (3D) with optional CBAM
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv3d -> BN -> ReLU) x 2, optionally followed by CBAM"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_cbam=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )
        self.cbam = CBAM3D(out_channels) if use_cbam else None

    def forward(self, x):
        x = self.double_conv(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x


class Down(nn.Module):
    """MaxPool3d then DoubleConv"""

    def __init__(self, in_channels, out_channels, use_cbam=False):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_channels, out_channels, use_cbam=use_cbam)

    def forward(self, x):
        return self.conv(self.maxpool(x))


class Up(nn.Module):
    """Upsample then DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=False, use_cbam=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_cbam=use_cbam)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_cbam=use_cbam)

    def forward(self, x1, x2):
        x1 = self.up(x1)

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

class UNet3D_CBAM(nn.Module):
    """
    3D U-Net with CBAM attention blocks.

    CBAM is inserted after every DoubleConv in both encoder and decoder.
    The bottleneck also gets CBAM.

    Args:
        n_channels:   input channels (1 for CT)
        n_classes:    output classes (2 for left/right parotid)
        input_shape:  (D, H, W) — each dim must be divisible by 16
        bilinear:     use trilinear upsampling instead of transposed convs
        base_filters: filters in first layer (64 recommended)
        cbam_ratio:   channel reduction ratio for CBAM MLP (default 16)
        cbam_kernel:  spatial attention kernel size (default 7)
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 2,
        input_shape: tuple = (48, 208, 272),
        bilinear: bool = False,
        base_filters: int = 64,
        cbam_ratio: int = 16,
        cbam_kernel: int = 7,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self._check_input_shape(input_shape)

        f = base_filters
        factor = 2 if bilinear else 1

        # encoder
        self.inc   = DoubleConv(n_channels, f,                use_cbam=True)
        self.down1 = Down(f,         f * 2,                   use_cbam=True)
        self.down2 = Down(f * 2,     f * 4,                   use_cbam=True)
        self.down3 = Down(f * 4,     f * 8,                   use_cbam=True)
        self.down4 = Down(f * 8,     f * 16 // factor,        use_cbam=True)  # bottleneck

        # decoder
        self.up1   = Up(f * 16,      f * 8  // factor, bilinear, use_cbam=True)
        self.up2   = Up(f * 8,       f * 4  // factor, bilinear, use_cbam=True)
        self.up3   = Up(f * 4,       f * 2  // factor, bilinear, use_cbam=True)
        self.up4   = Up(f * 2,       f,                bilinear, use_cbam=True)

        self.outc  = OutConv(f, n_classes)

    @staticmethod
    def _check_input_shape(shape):
        for i, s in enumerate(shape):
            if s % 16 != 0:
                raise ValueError(
                    f"Input dimension {i} is {s}, not divisible by 16. "
                    f"Nearest valid: {(s//16)*16} or {((s//16)+1)*16}."
                )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        logits = self.outc(x)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Guided Backpropagation (works for both UNet3D and UNet3D_CBAM)
# ---------------------------------------------------------------------------

class GuidedBackprop:
    """
    Guided backpropagation saliency maps.
    Works identically on vanilla UNet3D and UNet3D_CBAM — hooks attach
    to all ReLUs regardless of where they appear in the architecture.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(module.register_forward_hook(self._forward_hook))
                self.hooks.append(module.register_full_backward_hook(self._backward_hook))

    @staticmethod
    def _forward_hook(module, input, output):
        module._gbp_activation = output.detach()

    @staticmethod
    def _backward_hook(module, grad_input, grad_output):
        guided_grad = torch.clamp(grad_output[0], min=0.0)
        guided_grad = guided_grad * (module._gbp_activation > 0).float()
        return (guided_grad,)

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int = 0,
        target_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        self.model.eval()
        x = input_tensor.clone()
        x.requires_grad_(True)

        output = self.model(x)

        pred_channel = output[0, class_idx]
        if target_mask is not None:
            scalar = (pred_channel * target_mask.squeeze()).sum()
        else:
            foreground = (pred_channel > 0.5).float()
            scalar = (pred_channel * foreground).sum()

        self.model.zero_grad()
        scalar.backward()

        saliency = x.grad[0, 0].detach().cpu()
        saliency = torch.clamp(saliency, min=0.0)
        if saliency.max() > 0:
            saliency = saliency / saliency.max()

        return saliency.numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.ReLU) and hasattr(module, '_gbp_activation'):
                del module._gbp_activation


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    D, H, W = 48, 208, 272

    model = UNet3D_CBAM(
        n_channels=1,
        n_classes=2,
        input_shape=(D, H, W),
        bilinear=False,
        base_filters=64,
    )

    dummy = torch.randn(1, 1, D, H, W)
    out = model(dummy)

    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # GBP sanity check
    gbp = GuidedBackprop(model)
    saliency = gbp.generate(dummy, class_idx=0)
    gbp.remove_hooks()

    print(f"Saliency shape: {saliency.shape}")
    print(f"Saliency range: [{saliency.min():.3f}, {saliency.max():.3f}]")