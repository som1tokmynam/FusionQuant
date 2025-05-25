# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
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

# Install pyenv with better error handling
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
RUN echo "Creating flexible constraint file from main requirements.txt..." && \
    cp /home/user/app/requirements.txt /home/user/app/requirements-constraints.txt && \
    echo "Original requirements.txt (for constraints):" && \
    cat /home/user/app/requirements-constraints.txt && \
    echo "--- Processing constraints ---" && \
    # 1. Strip [extras] like [oauth]
    sed -i -E 's/^([^#\s]+)\[[^]]+\]/\1/' /home/user/app/requirements-constraints.txt && \
    # 2. Comment out git+ lines and lines with @ as they are not valid for pip constraint files
    sed -i -E '/^\s*git\+/s/^/# (Constraint-disabled VCS) /' /home/user/app/requirements-constraints.txt && \
    sed -i -E '/@/s/^/# (Constraint-disabled URL-based) /' /home/user/app/requirements-constraints.txt && \
    # 3. Convert strict version pins to minimum versions for flexibility (e.g. ==2.0.1 to >=2.0.1)
    #    Handle ==x.y.z, ==x.y
    sed -i -E 's/^([^#\s<>=!~]+)==([0-9]+\.[0-9]+(\.[0-9]+([a-zA-Z0-9.]*)?)?)/\1>=\2/' /home/user/app/requirements-constraints.txt && \
    echo "--- Processed constraint file (/home/user/app/requirements-constraints.txt) ---" && \
    cat /home/user/app/requirements-constraints.txt
USER 1000

# Setup llama.cpp by cloning the repository
# Temporarily switch to root to clone into APP_DIR which might be owned by user
USER root
RUN cd /home/user/app && \
    git clone https://github.com/ggerganov/llama.cpp.git && \
    chown -R 1000:1000 /home/user/app/llama.cpp # Ensure user owns the cloned repo
# Switch back to user
USER 1000

# Note: The llama.cpp compilation is now deferred to start.sh

# Install llama.cpp's Python requirements
# These are installed after the main requirements.txt to allow constraints to work,
# and then we'll fix huggingface-hub if it gets downgraded.
RUN cd /home/user/app/llama.cpp && \
    if [ -f requirements.txt ]; then \
        echo "Installing llama.cpp/requirements.txt" && \
        pip install --no-cache-dir -r requirements.txt --constraint /home/user/app/requirements-constraints.txt || \
        (echo "Constraint installation for llama.cpp/requirements.txt failed, trying without constraints..." && \
         pip install --no-cache-dir -r requirements.txt); \
    else \
        echo "llama.cpp/requirements.txt not found, skipping."; \
    fi

RUN cd /home/user/app/llama.cpp && \
    if [ -f requirements/requirements-convert_hf_to_gguf.txt ]; then \
        echo "Installing llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" && \
        pip install --no-cache-dir -r requirements/requirements-convert_hf_to_gguf.txt --constraint /home/user/app/requirements-constraints.txt || \
        (echo "Constraint installation for convert_hf_to_gguf.txt failed, trying without constraints..." && \
         pip install --no-cache-dir -r requirements/requirements-convert_hf_to_gguf.txt); \
    else \
        echo "llama.cpp/requirements/requirements-convert_hf_to_gguf.txt not found, skipping."; \
    fi

RUN cd /home/user/app/llama.cpp && \
    if [ -f requirements/requirements-tool_bench.txt ]; then \
        echo "Installing llama.cpp/requirements/requirements-tool_bench.txt" && \
        pip install --no-cache-dir -r requirements/requirements-tool_bench.txt --constraint /home/user/app/requirements-constraints.txt || \
        (echo "Constraint installation for tool_bench.txt failed, trying without constraints..." && \
         pip install --no-cache-dir -r requirements/requirements-tool_bench.txt); \
    else \
        echo "llama.cpp/requirements/requirements-tool_bench.txt not found, skipping."; \
    fi

# Force re-install/upgrade huggingface-hub to the version required by the main application
RUN echo "Ensuring correct huggingface-hub version for primary application dependencies..." && \
    pip install --no-cache-dir -U "huggingface-hub>=0.24.0,<1.0" && \
    echo "Final huggingface-hub version check:" && \
    pip show huggingface-hub

# Copy the rest of your application files
COPY --chown=user:user . /home/user/app/

# Copy groups_merged.txt if it exists into llama.cpp directory
# This should be done before start.sh runs, if start.sh's cmake process needs it
RUN if [ -f /home/user/app/groups_merged.txt ]; then \
        cp /home/user/app/groups_merged.txt /home/user/app/llama.cpp/groups_merged.txt; \
        echo "Copied groups_merged.txt to llama.cpp directory."; \
    else \
        echo "groups_merged.txt not found in app root, not copied."; \
    fi

# Copy examples for mergekit if they exist from the app root
RUN if [ -d /home/user/app/examples ]; then \
        mkdir -p /home/user/app/examples_target && \
        cp -r /home/user/app/examples/* /home/user/app/examples_target/ 2>/dev/null || echo "No examples to copy or error during copy." ; \
        echo "Copied mergekit examples."; \
    else \
        mkdir -p /home/user/app/examples_target; \
        echo "examples directory not found in app root, not copied."; \
    fi

# Create directories
RUN mkdir -p ${HOME}/app/outputs ${HOME}/app/downloads

# Copy the start.sh script and make it executable
COPY --chown=user:user start.sh /home/user/app/start.sh
RUN chmod +x /home/user/app/start.sh

# Environment variables
ENV PYTHONPATH=${HOME}/app \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/local/cuda/bin:${PATH} \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860" \
    GRADIO_THEME=huggingface \
    TQDM_POSITION=-1 \
    TQDM_MININTERVAL=1 \
    SYSTEM=spaces \
    RUN_LOCALLY=1 \
    PATH=${PATH}:${HOME}/.local/bin:${PYENV_ROOT}/shims:${PYENV_ROOT}/bin

# Expose Gradio port
EXPOSE 7860

# Entrypoint/CMD to run the start.sh script
CMD ["/bin/bash", "/home/user/app/start.sh"]
