# syntax=docker/dockerfile:1
# Single-stage dev image on CUDA 13.0.2 devel (Ubuntu 24.04)
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive

ARG CUDSS_VERSION=0.6.0.5
ARG CERES_BRANCH=93e66f0
ARG EIGEN_VERSION=3.4
ARG NANOFLANN_VERSION=v1.9.0
ARG COLMAP_VERSION=4.0.2

WORKDIR /app

# Base build deps
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    build-essential ninja-build git wget curl ca-certificates \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev \
    libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
    qt6-base-dev libqt6opengl6-dev libqt6openglwidgets6 libcgal-dev \
    libsuitesparse-dev libopenimageio-dev openimageio-tools \
    libatlas-base-dev libabsl-dev libmkl-full-dev \
    python3 python3-pip python3-dev python3-tk tk-dev \
    libboost-iostreams-dev libboost-serialization-dev libopencv-dev \
    libjpeg-dev libpng-dev libtiff-dev libxi-dev gcc-10 g++-10 libxinerama-dev \
    pybind11-dev libgoogle-glog0v6 libglew2.2 libcholmod5 \
    libatlas3-base libmetis5 libmkl-rt libqt6openglwidgets6 \
    libopencv-core406 libopencv-imgproc406 libopencv-imgcodecs406 \
    libopencv-calib3d406 libopencv-features2d406 libopencv-flann406 \
    libgomp1 libtbb12 libopengl0 libcudnn9-cuda-13 \
    libsuitesparse-dev libopenimageio-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib64:$LD_LIBRARY_PATH \
    CPATH=/usr/local/cuda/targets/x86_64-linux/include:$CPATH \
    PATH=/usr/local/cuda/bin:$PATH \
    NVDIFRAST_BACKEND=cuda

# CMake
RUN wget https://cmake.org/files/v3.31/cmake-3.31.4-linux-x86_64.tar.gz && \
    echo "e8c096779b25acf006190b79b42a09f350e6d3c33ab7d395ed86a0871a9f8e96  cmake-3.31.4-linux-x86_64.tar.gz" | sha256sum -c - && \
    tar -zxf cmake-3.31.4-linux-x86_64.tar.gz && \
    mv cmake-3.31.4-linux-x86_64 /opt/cmake-3.31.4 && \
    rm cmake-3.31.4-linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.31.4/bin:${PATH}"

# Install CUDSS (CUDA 13 archive)
RUN wget "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.7.1.4_cuda13-archive.tar.xz" && \
    mkdir -p /tmp/cudss && \
    tar -xJf libcudss-linux-x86_64-0.7.1.4_cuda13-archive.tar.xz -C /tmp/cudss --strip-components=1 && \
    cp -r /tmp/cudss/lib/ /usr/local/cuda/ && \
    cp -r /tmp/cudss/include/* /usr/local/cuda/include/ && \
    rm -rf /tmp/cudss libcudss-linux-x86_64-0.7.1.4_cuda13-archive.tar.xz


# Download and build Ceres Solver with CUDA support
WORKDIR /app
RUN git clone --recursive --shallow-submodules https://github.com/ceres-solver/ceres-solver.git && \
    cd ceres-solver && \
    git checkout ${CERES_BRANCH} && \
    mkdir build && cd build && \
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DCERES_USE_CUDA=ON -DCERES_USE_CUDSS=ON && \
    ninja -j$(nproc) && ninja install -j$(nproc) && \
    cd ../.. && rm -rf ceres-solver


# Python toolchain
ADD https://astral.sh/uv/0.10.12/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages --no-build-isolation --link-mode=copy \
    --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.10.0+cu130 torchvision==0.25.0+cu130


RUN TORCH_CUDA_ARCH_LIST="9.0" pip3 install --break-system-packages --no-build-isolation  git+https://github.com/pypose/pypose.git@07b2c23ae1f43d13397c6a2db6510f761375749e

RUN USE_CUDSS=1 TORCH_CUDA_ARCH_LIST="9.0" \
    CUDSS_DIR=/usr/local/cuda \
    LIBRARY_PATH=/usr/local/cuda/lib:$LIBRARY_PATH \
    pip3 install --break-system-packages --no-build-isolation -v \
    "bae @ git+https://github.com/sair-lab/bae.git@2e4284529c89a5100b3a42740744fcd3ad0a76bb"

RUN pip3 install --break-system-packages --no-build-isolation "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim"
ENV CFLAGS="-I/usr/include/suitesparse" \
    CXXFLAGS="-I/usr/include/suitesparse" \
    SUITESPARSE_INCLUDE_DIR=/usr/include/suitesparse \
    SUITESPARSE_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu
RUN pip3 install --break-system-packages scikit-sparse==0.4.15

WORKDIR /app

ENV CC=/usr/bin/gcc-10 \
    CXX=/usr/bin/g++-10 \
    CUDAHOSTCXX=/usr/bin/g++-10

# Build COLMAP with BAE enabled
COPY CMakeLists.txt /colmap/CMakeLists.txt
COPY cmake /colmap/cmake
COPY src /colmap/src
COPY python/CMakeLists.txt /colmap/python/CMakeLists.txt
COPY python/README.md /colmap/python/README.md
COPY python/pycolmap /colmap/python/pycolmap
COPY pyproject.toml /colmap/pyproject.toml
COPY doc/COLMAP.desktop /colmap/doc/COLMAP.desktop
WORKDIR /colmap
RUN mkdir -p build/.ccache && cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=90 -DONNX_ENABLED=ON -DBAE_ENABLED=ON && \
    ninja -j$(nproc) && ninja install -j$(nproc)

# Install pycolmap and copy solver (Python 3.12)
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN CMAKE_ARGS="-DGENERATE_STUBS=OFF" pip3 install -v . && \
    cp /colmap/python/pycolmap/bae_solver.py /usr/local/lib/python3.12/dist-packages/pycolmap/bae_solver.py


WORKDIR /app
