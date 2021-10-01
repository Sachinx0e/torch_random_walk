FROM determinedai/environments:cuda-11.1-pytorch-1.9-lightning-1.3-tf-2.4-gpu-0.16.4

# Install some basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ssh-client \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir -p /root/workdir && mkdir -p /root/.ssh && mkdir -p /root/.vscode-server && pip install ninja
WORKDIR /root/workdir

# Set the default command to python3
CMD ["bash"]