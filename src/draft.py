import torch
from unet_basic import UNet3D

model = UNet3D(n_channels=1, n_classes=2, input_shape=(48, 208, 272), base_filters=64)
model.eval()

dummy = torch.randn(1, 1, 48, 208, 272)
with torch.no_grad():
    out = model(dummy)
print(out.shape)