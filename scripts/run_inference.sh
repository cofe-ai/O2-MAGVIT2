#!/bin/bash
python inference.py \
--ckpt-path pytorch_model.bin \
--config-path configs/magvit2_3d_model_config.yaml \
--target-pixels 400 \
--max-num-frames 12 \
--extract-frame-interval 3 \
-i input.jpg \
-o tests \
--modal image