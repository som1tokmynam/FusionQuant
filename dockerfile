# Use Ubuntu base image instead of CUDA since we don't need GPU support
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    sed \
    cmake \
    sudo \
	file \
    # Python build dependencies for pyenv
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    # ffmpeg
    ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Define HOME and APP_DIR early
ENV HOME=/home/user
ENV APP_DIR=${HOME}/app

# Setup user properly - handle case where UID 1000 might already exist
RUN if id 1000 >/dev/null 2>&1; then \
        echo "UID 1000 already exists, using existing user"; \
        existing_user=$(id -nu 1000); \
        if [ "$existing_user" != "user" ]; then \
            usermod -l user $existing_user 2>/dev/null || echo "Could not rename user"; \
        fi; \
    else \
        useradd --create-home --uid 1000 --shell /bin/bash user; \
    fi && \
    mkdir -p ${APP_DIR} && \
    chown -R 1000:1000 ${HOME} && \
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the user (use UID since username might vary)
USER 1000

# Set WORKDIR; directory should now exist and be owned by 'user'
WORKDIR ${APP_DIR}

# Install pyenv and Python
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
ARG PYTHON_VERSION=3.11

# Install pyenv with better error handling and SSL certificate fix
RUN set -e && \
    echo "Installing pyenv..." && \
    curl -sSL https://pyenv.run | bash && \
    echo "pyenv installation completed" && \
    echo "Installing Python ${PYTHON_VERSION}..." && \
    ${PYENV_ROOT}/bin/pyenv install ${PYTHON_VERSION} && \
    echo "Setting global Python version..." && \
    ${PYENV_ROOT}/bin/pyenv global ${PYTHON_VERSION} && \
    ${PYENV_ROOT}/bin/pyenv rehash && \
    echo "Python installation completed"

# Upgrade pip and install common Python tools
RUN pip install --no-cache-dir -U pip setuptools wheel

# Install Python dependencies from the consolidated requirements.txt
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a more flexible constraint file that allows version ranges
USER root
RUN echo "Creating flexible constraint file..." && \
    cp /home/user/app/requirements.txt /home/user/app/requirements-constraints.txt && \
    echo "Original requirements.txt:" && \
    cat /home/user/app/requirements.txt && \
    echo "--- Processing constraints ---" && \
    # 1. Strip [extras] like [oauth]
    sed -i -E 's/^([^#\s]+)\[[^]]+\]/\1/' /home/user/app/requirements-constraints.txt && \
    # 2. Comment out git+ lines as they are not valid constraints
    sed -i -E '/^\s*git\+/s/^/# (Constraint-disabled VCS) /' /home/user/app/requirements-constraints.txt && \
    # 3. Convert strict version pins to minimum versions for flexibility
    sed -i -E 's/^([^#=]+)==([0-9]+\.[0-9]+(\.[0-9]+)?)/\1>=\2/' /home/user/app/requirements-constraints.txt && \
    # 4. Keep huggingface-hub flexible to resolve conflicts  <--- MODIFY THIS SECTION
    # Comment out or remove the following line:
    # sed -i -E 's/^huggingface-hub>=.*/# huggingface-hub - allowing flexible resolution/' /home/user/app/requirements-constraints.txt && \
    echo "--- Processed constraint file ---" && \
    cat /home/user/app/requirements-constraints.txt
USER 1000

# Setup llama.cpp
USER root
RUN cd /home/user/app && git clone https://github.com/ggerganov/llama.cpp.git && \
    chown -R 1000:1000 llama.cpp

# Build llama.cpp for CPU-only usage
USER root
RUN cd /home/user/app/llama.cpp && \
    echo "=== Building llama.cpp for CPU-only usage ===" && \
    echo "Current directory: $(pwd)" && \
    echo "=== CMAKE Configuration (CPU-only) ===" && \
    cmake -S . -B build \
        -DLLAMA_CURL=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_METAL=OFF \
        -DGGML_OPENCL=OFF \
        -DGGML_VULKAN=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_SERVER=OFF && \
    echo "=== Building llama.cpp components ===" && \
    cmake --build build --config Release --parallel $(nproc) && \
    echo "=== Build completed ===" && \
    echo "Contents of build directory:" && \
    find build -name "llama-*" -type f && \
    echo "Contents of build/bin directory:" && \
    ls -la build/bin/ 2>/dev/null || echo "build/bin directory does not exist" && \
    echo "Contents of build directory (all executables):" && \
    find build -type f -executable -name "llama-*" && \
    echo "=== Testing built executables ===" && \
    if [ -f build/bin/llama-quantize ]; then \
        echo "llama-quantize found in build/bin/"; \
        file build/bin/llama-quantize; \
    elif [ -f build/llama-quantize ]; then \
        echo "llama-quantize found in build/"; \
        file build/llama-quantize; \
    else \
        echo "llama-quantize not found, searching..."; \
        find build -name "*quantize*" -type f; \
    fi && \
    if [ -f build/bin/llama-imatrix ]; then \
        echo "llama-imatrix found in build/bin/"; \
        file build/bin/llama-imatrix; \
    elif [ -f build/llama-imatrix ]; then \
        echo "llama-imatrix found in build/"; \
        file build/llama-imatrix; \
    else \
        echo "llama-imatrix not found, searching..."; \
        find build -name "*imatrix*" -type f; \
    fi && \
    chown -R 1000:1000 /home/user/app/llama.cpp && \
    echo "llama.cpp built successfully for CPU-only usage"

USER 1000

# Install llama.cpp's requirements with more flexible approach
RUN cd /home/user/app/llama.cpp && \
    if [ -f requirements.txt ]; then \
        echo "Installing llama.cpp/requirements.txt with flexible resolution" && \
        # First try with constraints, if it fails, install without constraints
        pip install --no-cache-dir -r requirements.txt --constraint /home/user/app/requirements-constraints.txt || \
        (echo "Constraint installation failed, trying without constraints..." && \
         pip install --no-cache-dir -r requirements.txt) && \
        echo "Successfully installed llama.cpp/requirements.txt"; \
    else \
        echo "llama.cpp/requirements.txt not found, skipping."; \
    fi

# Install convert_hf_to_gguf requirements with flexible approach
RUN cd /home/user/app/llama.cpp && \
    if [ -f requirements/requirements-convert_hf_to_gguf.txt ]; then \
        echo "Installing llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" && \
        pip install --no-cache-dir -r requirements/requirements-convert_hf_to_gguf.txt --constraint /home/user/app/requirements-constraints.txt || \
        (echo "Constraint installation failed, trying without constraints..." && \
         pip install --no-cache-dir -r requirements/requirements-convert_hf_to_gguf.txt) && \
        echo "Successfully installed requirements-convert_hf_to_gguf.txt"; \
    else \
        echo "llama.cpp/requirements/requirements-convert_hf_to_gguf.txt not found, skipping."; \
    fi

# Install tool_bench requirements with flexible approach
RUN cd /home/user/app/llama.cpp && \
    if [ -f requirements/requirements-tool_bench.txt ]; then \
        echo "Installing llama.cpp/requirements/requirements-tool_bench.txt" && \
        pip install --no-cache-dir -r requirements/requirements-tool_bench.txt --constraint /home/user/app/requirements-constraints.txt || \
        (echo "Constraint installation failed, trying without constraints..." && \
         pip install --no-cache-dir -r requirements/requirements-tool_bench.txt) && \
        echo "Successfully installed requirements-tool_bench.txt"; \
    else \
        echo "llama.cpp/requirements/requirements-tool_bench.txt not found, skipping."; \
    fi

# Copy the rest of your application files
COPY --chown=user:user . /home/user/app/

# Copy groups_merged.txt if it exists
RUN if [ -f groups_merged.txt ]; then \
        cp groups_merged.txt ${HOME}/app/llama.cpp/groups_merged.txt; \
    fi

# Copy examples for mergekit if they exist
RUN if [ -d examples ]; then \
        cp -r examples/* ./examples/ 2>/dev/null || mkdir -p ./examples; \
    fi

# Create directories
RUN mkdir -p ${HOME}/app/outputs ${HOME}/app/downloads

# Environment variables (removed CUDA-related vars)
ENV PYTHONPATH=${HOME}/app \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860" \
    GRADIO_THEME=huggingface \
    TQDM_POSITION=-1 \
    TQDM_MININTERVAL=1 \
    SYSTEM=spaces \
    PATH=${PATH}:${HOME}/.local/bin:${PYENV_ROOT}/shims:${PYENV_ROOT}/bin

# Expose Gradio port
EXPOSE 7860

# Entrypoint/CMD
CMD ["python", "combined_app.py"]