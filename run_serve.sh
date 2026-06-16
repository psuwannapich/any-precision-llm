#!/bin/bash
module load compiler/GCC/13.2.0 system/CUDA/12.6.0
TORCH_LIB=/mnt/aiongpfs/users/psuwannapichat/work_space/any-precision-llm/.venv/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH=$TORCH_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
cd /mnt/aiongpfs/users/psuwannapichat/work_space/any-precision-llm
.venv/bin/python serve.py "$@"
