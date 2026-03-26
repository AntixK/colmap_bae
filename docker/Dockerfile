# syntax=docker/dockerfile:1

ARG UBUNTU_VERSION=24.04
ARG NVIDIA_CUDA_VERSION=12.9.1

#
# Docker builder stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG CUDA_ARCHITECTURES=all-major
ENV QT_XCB_GL_INTEGRATION=xcb_egl
ARG FETCHCONTENT_FULLY_DISCONNECTED=OFF
ENV CCACHE_DIR=/colmap/build/.ccache
ENV CCACHE_BASEDIR=/colmap

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y \
    ccache \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libopenimageio-dev \
    openimageio-tools \
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
    libceres-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libmkl-full-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    wget

# Install BAE Python dependencies.
RUN pip3 install --break-system-packages --no-build-isolation \
    torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --break-system-packages --no-build-isolation \
    scipy tqdm warp-lang==1.5.1 torchdata==0.8.0 "portalocker>=2.0.0" && \
    pip3 install --break-system-packages --no-build-isolation \
    git+https://github.com/pypose/pypose.git@bae
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Install cuDSS.
ARG CUDSS_VERSION=0.6.0.5
RUN wget "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz" && \
    echo "159ce1d4e3e4bba13b0bd15cf943e44b869c53b7a94f9bac980768c927f02e75  libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz" | sha256sum -c - && \
    mkdir -p /tmp/cudss && \
    tar -xJf libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz -C /tmp/cudss --strip-components=1 && \
    cp -r /tmp/cudss/lib/ /usr/local/cuda/ && \
    cp -r /tmp/cudss/include/* /usr/local/cuda/include/ && \
    mkdir -p /usr/lib/x86_64-linux-gnu/libcudss/12 && \
    ln -sf /usr/local/cuda/lib/libcudss* /usr/lib/x86_64-linux-gnu/libcudss/12/ && \
    mkdir -p /usr/include/libcudss/12 && \
    for f in /usr/local/cuda/include/cudss*; do ln -sf "$f" /usr/include/libcudss/12/; done && \
    rm -rf /tmp/cudss libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz

# Install BAE with CUDSS support.
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
ENV CUDSS_DIR=/usr/local/cuda
RUN USE_CUDSS=1 pip3 install --break-system-packages --no-build-isolation -v \
    "bae @ git+https://github.com/sair-lab/bae.git@2e4284529c89a5100b3a42740744fcd3ad0a76bb"

# Fix issue in Ubuntu's openimageio CMake config.
# We don't depend on any of openimageio's OpenCV functionality,
# but it still requires the OpenCV include directory to exist.
RUN mkdir -p /usr/include/opencv4

# Copy source into the image.
COPY . /colmap

# Build and install COLMAP.
RUN cd /colmap && \
    mkdir -p build/.ccache && \
    cd build && \
    cmake .. \
    -GNinja \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    -DCMAKE_INSTALL_PREFIX=/colmap-install \
    -DFETCHCONTENT_FULLY_DISCONNECTED=${FETCHCONTENT_FULLY_DISCONNECTED} \
    -DBLA_VENDOR=Intel10_64lp \
    -DBAE_ENABLED=ON && \
    ninja install

#
# Stage to export build caches for CI round-tripping via actions/cache.
#
FROM scratch AS cache-export
COPY --from=builder /colmap/build/.ccache/ /.ccache/
COPY --from=builder /colmap/build/_deps/ /_deps/

#
# Docker runtime stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# Minimal dependencies to run COLMAP binary compiled in the builder stage.
# Note: this reduces the size of the final image considerably, since all the
# build dependencies are not needed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
    libboost-program-options1.83.0 \
    libc6 \
    libomp5 \
    libopengl0 \
    libmetis5 \
    libceres4t64 \
    libopenimageio2.4t64 \
    libgcc-s1 \
    libgl1 \
    libglew2.2 \
    libgoogle-glog0v6t64 \
    libqt6core6 \
    libqt6gui6 \
    libqt6widgets6 \
    libqt6openglwidgets6 \
    libcurl4 \
    libssl3t64 \
    libmkl-locale \
    libmkl-intel-lp64 \
    libmkl-intel-thread \
    libmkl-core && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all files from /colmap-install/ in the builder stage to /usr/local/ in
# the runtime stage. This simulates installing COLMAP in the default location
# (/usr/local/), which simplifies environment variables. It also allows the user
# of this Docker image to use it as a base image for compiling against COLMAP as
# a library. For instance, CMake will be able to find COLMAP easily with the
# command: find_package(COLMAP REQUIRED).
# Install Python and BAE dependencies in runtime stage.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
    python3 \
    python3-pip \
    python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy CUDA libraries needed by Warp (NVRTC, nvJitLink) from the builder.
# The runtime base image only includes libcudart; Warp also needs these to
# compile and run CUDA kernels via the driver API.
COPY --from=builder /usr/local/cuda/lib64/libnvrtc* /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda/lib64/libnvJitLink* /usr/local/cuda/lib64/

# Copy Python packages from builder.
COPY --from=builder /usr/lib/python3/dist-packages/ /usr/lib/python3/dist-packages/
COPY --from=builder /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/

COPY --from=builder /colmap-install/ /usr/local/
