#!/bin/bash

set -ex

# install pytorch
pip install numpy tensorboard typing-extensions pillow pytest
if [ "$PYTORCH_VERSION" = "nightly" ]; then
    pip install --pre torch -f "https://download.pytorch.org/whl/nightly/$CUDA_VERSION/torch_nightly.html"
    pip install --pre torchvision --no-deps -f "https://download.pytorch.org/whl/nightly/$CUDA_VERSION/torch_nightly.html"
elif [ "$PYTORCH_VERSION" = "1.11rc" ]; then
    pip install --pre torch -f "https://download.pytorch.org/whl/test/$CUDA_VERSION/torch_test.html"
    pip install --pre torchvision --no-deps -f "https://download.pytorch.org/whl/test/$CUDA_VERSION/torch_test.html"
elif [ "$PYTORCH_VERSION" = "stable" ]; then
    pip install torch torchvision
fi

python -c "import torch; print(torch.__version__, torch.version.git_version); from torch.autograd import kineto_available; print(kineto_available())"
