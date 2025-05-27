# Step 1: Use your pre-compiled custom base image
# Replace 'my-precompiled-app-base:v1' with the actual name and tag you used
FROM som1tokmynam/precompiled-base:1.0

# Inherited from base:
# - User 'builder' (UID 1000)
# - WORKDIR /home/builder/app (or similar, ensure consistency)
# - Python environment with all dependencies from the base's requirements.txt
# - Compiled llama.cpp tools in PATH
# - Compiled exllamav2 extensions
# - ENV VARS like CUDA_HOME, PATH, LD_LIBRARY_PATH, PYTHONUNBUFFERED, HF_HUB_ENABLE_HF_TRANSFER,
#   LLAMA_CPP_DIR, TORCH_CUDA_ARCH_LIST, PYTHONPATH

# Set the working directory for the application (should match what was set in the base for consistency)
# The base Dockerfile set WORKDIR ${APP_DIR} which is /home/builder/app
WORKDIR /home/builder/app

# Copy your application files into the working directory
COPY --chown=builder:builder combined_app.py ./
COPY --chown=builder:builder gguf_utils.py ./
COPY --chown=builder:builder mergekit_utils.py ./
COPY --chown=builder:builder exllamav2_utils.py ./
COPY --chown=builder:builder groups_merged.txt ./
COPY examples/ ./examples/

# Ensure groups_merged.txt is in the llama.cpp source directory if your scripts expect it there.
# LLAMA_CPP_DIR is /home/builder/app/llama.cpp (set in the base image).
RUN if [ -f "./groups_merged.txt" ]; then \
        # Check if the target directory exists, create if not (though base should have cloned llama.cpp)
        mkdir -p "${LLAMA_CPP_DIR}" && \
        cp "./groups_merged.txt" "${LLAMA_CPP_DIR}/groups_merged.txt"; \
        echo "Copied groups_merged.txt to ${LLAMA_CPP_DIR} for the application."; \
    else \
        echo "groups_merged.txt not found in application source, not copied to llama.cpp dir."; \
    fi

# Create output/download directories if your app needs them and they aren't created in the base image.
# The base image's user 'builder' should own APP_DIR, so this should work.
RUN mkdir -p ./outputs ./downloads

# Application-specific environment variables for Gradio, etc.
# Many common ones are already set in the base image.
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV GRADIO_THEME=huggingface          
ENV GRADIO_ALLOW_FLAGGING=never
ENV TQDM_POSITION=-1
ENV TQDM_MININTERVAL=1
ENV SYSTEM=spaces
ENV RUN_LOCALLY=1

# Expose the Gradio port
EXPOSE 7860

# Set the entrypoint to run your main application script
# This assumes combined_app.py is now in the WORKDIR (/home/builder/app)
ENTRYPOINT ["python3", "combined_app.py"]

# Default command (can be empty if ENTRYPOINT is self-contained)
CMD []