# Use ghcr.io/ggml-org/llama.cpp base image
FROM ghcr.io/ggerganov/llama.cpp:full-cuda-b4719

ENV DEBIAN_FRONTEND=noninteractive
# Install fewer packages, as the new base image (likely based on nvidia/cuda:<ver>-devel)
# should have git, curl, wget, build-essential, python3, pip, cmake etc.
# Keeping git-lfs, sudo (for user setup), ffmpeg, libcurl (for some python packages), sed (used later).
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
    sudo \
    file \
    ffmpeg \
    python3 \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Define HOME and APP_DIR early
ENV HOME=/home/user
ENV APP_DIR=${HOME}/app

# Setup user properly - handle case where UID 1000 might already exist [cite: 3]
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

# Use system Python from the base image. Upgrade pip and install common Python tools.
# The base image ghcr.io/ggml-org/llama.cpp:server-cuda-b5478 should provide Python 3 and pip.
RUN python3 -m pip install --no-cache-dir --user -U pip setuptools wheel

# Add user's local bin to PATH for executables installed by pip install --user
ENV PATH="${HOME}/.local/bin:${PATH}"

# Install Python dependencies from the consolidated requirements.txt
# pip will use the system python and install packages for the current user (1000)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Create a more flexible constraint file that allows version ranges
# Switching to root temporarily for this operation as it copies from potentially root-owned source in original Dockerfile context
# However, requirements.txt is now copied as 'user', so root might not be needed if APP_DIR is user-owned.
# For safety and consistency with original, using root for cp, then revert.
USER root
RUN echo "Creating flexible constraint file from main requirements.txt..." && \
    cp /home/user/app/requirements.txt /home/user/app/requirements-constraints.txt && \
    echo "Original requirements.txt (for constraints):" && \
    cat /home/user/app/requirements-constraints.txt && \
    echo "--- Processing constraints ---" && \
    sed -i -E 's/^([^#\s]+)\[[^]]+\]/\1/' /home/user/app/requirements-constraints.txt && \
    sed -i -E '/^\s*git\+/s/^/# (Constraint-disabled VCS) /' /home/user/app/requirements-constraints.txt && \
    sed -i -E '/@/s/^/# (Constraint-disabled URL-based) /' /home/user/app/requirements-constraints.txt && \
    sed -i -E 's/^([^#\s<>=!~]+)==([0-9]+\.[0-9]+(\.[0-9]+([a-zA-Z0-9.]*)?)?)/\1>=\2/' /home/user/app/requirements-constraints.txt && \
    echo "--- Processed constraint file (/home/user/app/requirements-constraints.txt) ---" && \
    cat /home/user/app/requirements-constraints.txt && \
    chown 1000:1000 /home/user/app/requirements-constraints.txt
USER 1000

# Setup llama.cpp by cloning the repository
# This is kept for Python scripts and expected directory structure by gguf_utils.py,
# Base image provides compiled llama.cpp tools.
USER root
RUN cd /home/user/app && \
    git clone https://github.com/ggerganov/llama.cpp.git && \
    chown -R 1000:1000 /home/user/app/llama.cpp
USER 1000

# Install llama.cpp's Python requirements
# These are for the Python scripts within the cloned llama.cpp repo.
RUN cd /home/user/app/llama.cpp && \
    if [ -f requirements.txt ]; then \
        echo "Installing llama.cpp/requirements.txt" && \
        pip install --no-cache-dir --user -r requirements.txt --constraint /home/user/app/requirements-constraints.txt || \
        (echo "Constraint installation for llama.cpp/requirements.txt failed, trying without constraints..." && \
         pip install --no-cache-dir --user -r requirements.txt); \
    else \
        echo "llama.cpp/requirements.txt not found, skipping."; \
    fi

RUN cd /home/user/app/llama.cpp && \
    if [ -f requirements/requirements-convert_hf_to_gguf.txt ]; then \
        echo "Installing llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" && \
        pip install --no-cache-dir --user -r requirements/requirements-convert_hf_to_gguf.txt --constraint /home/user/app/requirements-constraints.txt || \
        (echo "Constraint installation for convert_hf_to_gguf.txt failed, trying without constraints..." && \
         pip install --no-cache-dir --user -r requirements/requirements-convert_hf_to_gguf.txt); \
    else \
        echo "llama.cpp/requirements/requirements-convert_hf_to_gguf.txt not found, skipping."; \
    fi

RUN cd /home/user/app/llama.cpp && \
    if [ -f requirements/requirements-tool_bench.txt ]; then \
        echo "Installing llama.cpp/requirements/requirements-tool_bench.txt" && \
        pip install --no-cache-dir --user -r requirements/requirements-tool_bench.txt --constraint /home/user/app/requirements-constraints.txt || \
        (echo "Constraint installation for tool_bench.txt failed, trying without constraints..." && \
         pip install --no-cache-dir --user -r requirements/requirements-tool_bench.txt); \
    else \
        echo "llama.cpp/requirements/requirements-tool_bench.txt not found, skipping."; \
    fi

# Force re-install/upgrade huggingface-hub to the version required by the main application
RUN echo "Ensuring correct huggingface-hub version for primary application dependencies..." && \
    pip install --no-cache-dir --user -U "huggingface-hub>=0.24.0,<1.0" && \
    echo "Final huggingface-hub version check:" && \
    pip show huggingface-hub

# Copy the rest of your application files
COPY --chown=user:user . /home/user/app/

# Copy groups_merged.txt if it exists into llama.cpp directory
RUN if [ -f /home/user/app/groups_merged.txt ]; then \
        cp /home/user/app/groups_merged.txt /home/user/app/llama.cpp/groups_merged.txt; \
        echo "Copied groups_merged.txt to llama.cpp directory."; \
    else \
        echo "groups_merged.txt not found in app root, not copied."; \
    fi

# Create directories
RUN mkdir -p ${HOME}/app/outputs ${HOME}/app/downloads

# Environment variables
# Removed PYENV_ROOT from PATH. Added ${HOME}/.local/bin for user pip installs.
# CUDA_HOME, LD_LIBRARY_PATH, and CUDA PATH are kept; the base image might set them,
# but these are standard paths and should be compatible or correctly appended.
ENV PYTHONPATH=${HOME}/app \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/local/cuda/bin:${HOME}/.local/bin:${PATH} \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860" \
    GRADIO_THEME=huggingface \
    TQDM_POSITION=-1 \
    TQDM_MININTERVAL=1 \
    SYSTEM=spaces \
    RUN_LOCALLY=1

# Expose Gradio port
EXPOSE 7860
# Set the entrypoint to directly execute the python script
# WORKDIR is /home/user/app, so combined_app.py should be found
ENTRYPOINT ["python3", "combined_app.py"]

# You can remove the old CMD or set it to empty if no default arguments are needed for combined_app.py
CMD []