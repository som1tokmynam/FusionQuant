# Step 1: Use your pre-compiled custom base image
FROM som1tokmynam/precompiled-base:latest

# Set the working directory for the application
WORKDIR /home/builder/app

# Copy your application files into the working directory.
# The --chown flag is great because the Docker daemon (which is root) handles it.
COPY --chown=builder:builder combined_app.py ./
COPY --chown=builder:builder gguf_utils.py ./
COPY --chown=builder:builder mergekit_utils.py ./
COPY --chown=builder:builder exllamav2_utils.py ./
COPY --chown=builder:builder groups_merged.txt ./
# For the directory, we'll fix ownership in a RUN step as it's cleaner.
COPY examples/ ./examples/

# --- START OF ROOT-LEVEL OPERATIONS ---
# Explicitly switch to the root user to perform system-level tasks
USER root

# Run the script copy to llama.cpp dir
RUN if [ -f "./groups_merged.txt" ]; then \
        mkdir -p "${LLAMA_CPP_DIR}" && \
        cp "./groups_merged.txt" "${LLAMA_CPP_DIR}/groups_merged.txt"; \
        echo "Copied groups_merged.txt to ${LLAMA_CPP_DIR} for the application."; \
    else \
        echo "groups_merged.txt not found in application source, not copied to llama.cpp dir."; \
    fi

# Now, as root, create directories AND fix ownership of the copied 'examples' directory.
RUN mkdir -p ./outputs ./downloads && \
    chown -R builder:builder ./examples ./outputs ./downloads

# Copy startup scripts and make them executable (requires root)
COPY startup.sh /usr/local/bin/startup.sh
COPY entrypoint_wrapper.sh /usr/local/bin/entrypoint_wrapper.sh
RUN chmod +x /usr/local/bin/startup.sh /usr/local/bin/entrypoint_wrapper.sh

# --- END OF ROOT-LEVEL OPERATIONS ---

# Application-specific environment variables for Gradio, etc.
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
# ... (other ENVs are fine) ...

# Expose the Gradio port
EXPOSE 7860

# BEST PRACTICE: Switch back to the non-root user before setting the ENTRYPOINT.
# This ensures the container runs as the 'builder' user.
USER builder

# Set the new entrypoint to be the wrapper script
ENTRYPOINT ["/usr/local/bin/entrypoint_wrapper.sh"]

# Default command
CMD []