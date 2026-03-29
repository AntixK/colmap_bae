# syntax=docker/dockerfile:1
# The above line enables buildkit
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive

# ------------------------------
ARG CUDSS_VERSION=0.6.0.5 # Note: You have to change the checksum as well
ARG CERES_BRANCH=93e66f0
ARG EIGEN_VERSION=3.4
ARG NANOFLANN_VERSION=v1.9.0
ARG COLMAP_VERSION=4.0.2
# ARG OPENMVS_VERSION=v2.4.0 # TODO: Move to official openMVS repo
# -----------------------------
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
    build-essential \
    ninja-build \
    git \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qt6-base-dev \
    libqt6opengl6-dev \
    libqt6openglwidgets6 \
    libcgal-dev \
    libsuitesparse-dev \
    libopenimageio-dev \
    openimageio-tools \
    libatlas-base-dev \
    libabsl-dev \
    libmkl-full-dev \
    wget \
    curl \
    python3-pip \
    python3-tk \
    python3-dev \
    tk-dev \
    libboost-iostreams-dev \
    libboost-serialization-dev \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libxi-dev \
    gcc-10 g++-10 \
    libxinerama-dev && \
    rm -rf /var/lib/apt/lists/*


RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib64:$LD_LIBRARY_PATH \
    CPATH=/usr/local/cuda/targets/x86_64-linux/include:$CPATH \
    PATH=/usr/local/cuda/bin:$PATH \
    NVDIFRAST_BACKEND=cuda


# Install CMake
RUN wget https://cmake.org/files/v3.31/cmake-3.31.4-linux-x86_64.tar.gz && \
    echo "e8c096779b25acf006190b79b42a09f350e6d3c33ab7d395ed86a0871a9f8e96  cmake-3.31.4-linux-x86_64.tar.gz" | sha256sum -c - && \
    tar -zxf cmake-3.31.4-linux-x86_64.tar.gz && \
    mv cmake-3.31.4-linux-x86_64 /opt/cmake-3.31.4 && \
    rm cmake-3.31.4-linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.31.4/bin:${PATH}"

# Install CUDSS
RUN wget "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz" && \
    echo "159ce1d4e3e4bba13b0bd15cf943e44b869c53b7a94f9bac980768c927f02e75  libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz" | sha256sum -c - && \
    mkdir -p /tmp/cudss && \
    tar -xJf libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz -C /tmp/cudss --strip-components=1 && \
    cp -r /tmp/cudss/lib/ /usr/local/cuda/ && \
    cp -r /tmp/cudss/include/* /usr/local/cuda/include/ && \
    rm -rf /tmp/cudss libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz


# Download and build Ceres Solver with CUDA support
WORKDIR /app
RUN git clone --recursive --shallow-submodules https://github.com/ceres-solver/ceres-solver.git && \
    cd ceres-solver && \
    git checkout ${CERES_BRANCH} && \
    mkdir build && \
    cd build && \
    cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DCERES_USE_CUDA=ON \
    -DCERES_USE_CUDSS=ON && \
    ninja -j$(nproc) && \
    ninja install -j$(nproc) && \
    cd ../.. && \
    rm -rf ceres-solver

# Use GCC 10 for COLMAP > 4.0
ENV CC=/usr/bin/gcc-10 \
    CXX=/usr/bin/g++-10 \
    CUDAHOSTCXX=/usr/bin/g++-10



# Install pybind11-dev for BAE's pybind11::embed
RUN apt-get update && apt-get install -y --no-install-recommends pybind11-dev && \
    rm -rf /var/lib/apt/lists/*

# Install COLMAP (with BAE enabled)
WORKDIR /app
COPY . /colmap
RUN cd /colmap && \
    mkdir -p build/.ccache && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native -DONNX_ENABLED=ON -DBAE_ENABLED=ON && \
    ninja -j$(nproc) && \
    ninja install -j$(nproc)

# Build and install pycolmap Python bindings, then inject bae_solver.py
RUN cd /colmap && \
    CMAKE_ARGS="-DGENERATE_STUBS=OFF" pip3 install -v . && \
    cp /colmap/python/pycolmap/bae_solver.py \
    /usr/local/lib/python3.10/dist-packages/pycolmap/bae_solver.py && \
    rm -rf /colmap


# Python environment setup
# Download the latest installer
ADD https://astral.sh/uv/0.10.12/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-build-isolation --link-mode=copy \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124


RUN TORCH_CUDA_ARCH_LIST="8.6" pip3 install --no-build-isolation  git+https://github.com/pypose/pypose.git@07b2c23ae1f43d13397c6a2db6510f761375749e

RUN USE_CUDSS=1 TORCH_CUDA_ARCH_LIST="8.6" \
    CUDSS_DIR=/usr/local/cuda \
    LIBRARY_PATH=/usr/local/cuda/lib:$LIBRARY_PATH \
    pip3 install --no-build-isolation -v \
    "bae @ git+https://github.com/sair-lab/bae.git@2e4284529c89a5100b3a42740744fcd3ad0a76bb"

# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv pip install --system --link-mode=copy xformers -i https://download.pytorch.org/whl/cu124

WORKDIR /app
# RUN git clone --recursive --shallow-submodules https://github.com/Anand-EB/Hierarchical-Localization
# RUN --mount=type=cache,target=/root/.cache/uv \
#     cd Hierarchical-Localization && \
#     uv pip install --system -e . --no-build-isolation --link-mode=copy && \
#     find /app/Hierarchical-Localization -name .git -type d -exec rm -rf {} + 2>/dev/null || true


# --- Final Stage ---
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# FROM nvidia/cuda:12.4.0-base-ubuntu22.04


ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Enable repositories and prepare a clean layer
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common curl gpg && \
    add-apt-repository -y universe && \
    add-apt-repository -y multiverse && \
    add-apt-repository -y restricted && \
    # Manual PPA add to avoid Launchpad timeout
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF4E48910A020E77056748B745738AE8480447DDF" | gpg --dearmor -o /usr/share/keyrings/ubuntuhandbook1-ffmpeg8.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ubuntuhandbook1-ffmpeg8.gpg] https://ppa.launchpadcontent.net/ubuntuhandbook1/ffmpeg8/ubuntu jammy main" > /etc/apt/sources.list.d/ffmpeg8.list && \
    apt-get update && \
    apt-get purge -y software-properties-common curl gpg && \
    rm -rf /var/lib/apt/lists/*

# Install minimal runtime dependencies after a clean layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime libs
    libboost-program-options1.74.0 \
    libboost-iostreams1.74.0 \
    libboost-serialization1.74.0 \
    libfreeimage3 \
    libgoogle-glog0v5 \
    libsqlite3-0 \
    libglew2.2 \
    libcholmod3 \
    libspqr2 \
    libsuitesparseconfig5 \
    libatlas3-base \
    libmetis5 \
    libopenimageio2.2 \
    libmkl-rt \
    libqt6openglwidgets6 \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    libopencv-calib3d4.5d \
    libopencv-features2d4.5d \
    libopencv-flann4.5d \
    libgomp1 \
    libtbb2 \
    libopengl0 \
    libcudnn9-cuda-12 \
    # Non-library packages (build tools kept per request)
    python3 \
    ffmpeg \
    python3-tk \
    python3-pip \
    git \
    build-essential \
    python3.10-dev \
    exiftool \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Copy artifacts from builder
COPY --from=builder /usr/local /usr/local

# Python setup
RUN ln -s /usr/bin/python3 /usr/bin/python

# Environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVDIFRAST_BACKEND=cuda