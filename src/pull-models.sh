#!/bin/bash
pip install -r requirements.txt
pip install huggingface_hub

# pull model weights
mkdir -p models
huggingface-cli download leo7d3/unet unet_20260609_150324_n52_lr0.0001_f64.pt --local-dir models/
huggingface-cli download leo7d3/unet cbam_20260609_162153_n52_lr0.0001_f64.pt --local-dir models/
