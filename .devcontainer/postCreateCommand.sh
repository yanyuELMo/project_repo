#!/usr/bin/env bash
set -e

bash -lc "
  conda activate mlops

  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

  python -c \"import torch; print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); \
print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')\"
"
