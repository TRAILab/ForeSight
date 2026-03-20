FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Install the MMDetection3D required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install StreamPETR required packages
WORKDIR /workspace/ForeSight/
COPY requirement.txt .
COPY projects/ projects/
RUN pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html
RUN pip install --no-cache-dir -r requirement.txt
RUN cd projects/mmdet3d_plugin/ops && python setup.py develop
RUN git config --global --add safe.directory '*'

WORKDIR /workspace/ForeSight/