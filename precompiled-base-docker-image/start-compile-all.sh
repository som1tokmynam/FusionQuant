#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
echo "===== Runtime Compilation and Startup Script ====="

# Environment variables should be inherited from the Docker image
APP_DIR="${HOME_DIR}/app"
LLAMA_CPP_DIR="${APP_DIR}/llama.cpp"
LOCAL_BIN_DIR="${HOME_DIR}/.local/bin"

# Ensure .local/bin (for llama.cpp tools) and CUDA bins are in PATH
export PATH="${LOCAL_BIN_DIR}:${CUDA_HOME}/bin:${PATH}"
# Ensure user site-packages are findable by Python
export PYTHONPATH="${HOME_DIR}/.local/lib/python3.10/site-packages:${PYTHONPATH}"

# Marker files to check if compilation has already been done
EXLLAMA_V2_COMPILED_MARKER="${APP_DIR}/.exllamav2_compiled"
LLAMA_CPP_COMPILED_MARKER="${LLAMA_CPP_DIR}/build/.compiled_with_cuda"

# --- 1. Compile ExLlamaV2 Extensions ---
if [ ! -f "${EXLLAMA_V2_COMPILED_MARKER}" ]; then
    echo "INFO: Attempting to compile ExLlamaV2 CUDA extensions at runtime..."
    python3 -c "import exllamav2; from exllamav2.ext import exllamav2_ext; print(f'ExLlamaV2 (v{exllamav2.__version__}) extensions module loaded: {exllamav2_ext}')"
    if [ $? -eq 0 ]; then
        echo "INFO: ExLlamaV2 extensions compiled/loaded successfully."
        mkdir -p "$(dirname "${EXLLAMA_V2_COMPILED_MARKER}")"
        touch "${EXLLAMA_V2_COMPILED_MARKER}"
    else
        echo "ERROR: Failed to compile/load ExLlamaV2 extensions at runtime."
        exit 1
    fi
else
    echo "INFO: ExLlamaV2 extensions already compiled (marker found)."
    python3 -c "import exllamav2; from exllamav2.ext import exllamav2_ext; print(f'ExLlamaV2 extensions loaded: {exllamav2_ext}')" || \
        (echo "ERROR: Failed to load previously compiled ExLlamaV2 extensions." && exit 1)
fi
echo "-----------------------------------------------------"

# --- 2. Compile llama.cpp ---
if [ ! -f "${LLAMA_CPP_COMPILED_MARKER}" ] || [ ! -x "${LOCAL_BIN_DIR}/llama-quantize" ]; then
    echo "INFO: llama.cpp not compiled with CUDA or tools not found in PATH. Compiling now..."
    cd "${LLAMA_CPP_DIR}"
    if [ -d "build" ]; then
        echo "INFO: Removing existing llama.cpp build directory..."
        rm -rf "build"
    fi
    mkdir -p "build"
    cd "build"

    echo "INFO: Running CMake for llama.cpp with CUDA..."
    # MODIFICATION: Ensure LLAMA_BUILD_EXAMPLES=ON if you need 'main' and other tools
    # Remove LLAMA_BUILD_SERVER=ON if you don't need the server executable specifically
    # Standard tools like quantize, perplexity, embedding are often built by default or with examples.
    cmake .. -DLLAMA_CUDA=ON -DLLAMA_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF \
             -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON \
             -DLLAMA_SERVER=OFF # Explicitly turn off server if not needed, ON if needed.
                                # For "full cuda" tools, LLAMA_BUILD_EXAMPLES=ON is usually sufficient.

    NPROC=$(nproc --ignore=1)
    echo "INFO: Building llama.cpp with ${NPROC} cores..."
    cmake --build . --config Release -- -j${NPROC}
    echo "INFO: llama.cpp compiled successfully."

    echo "INFO: Copying llama.cpp executables to ${LOCAL_BIN_DIR}..."
    mkdir -p "${LOCAL_BIN_DIR}"
    # Check for existence before copying, as not all tools are guaranteed with every build config
    if [ -f "bin/main" ]; then cp "bin/main" "${LOCAL_BIN_DIR}/llama-cli"; else echo "WARNING: bin/main not found."; fi
    if [ -f "bin/quantize" ]; then cp "bin/quantize" "${LOCAL_BIN_DIR}/llama-quantize"; else echo "WARNING: bin/quantize not found."; fi
    if [ -f "bin/server" ] && [ "${LLAMA_SERVER}" == "ON" ]; then cp "bin/server" "${LOCAL_BIN_DIR}/llama-server"; else echo "INFO: bin/server not built or not requested."; fi
    if [ -f "bin/gguf-split" ]; then cp "bin/gguf-split" "${LOCAL_BIN_DIR}/llama-gguf-split"; else echo "WARNING: bin/gguf-split not found."; fi
    if [ -f "bin/imatrix" ]; then cp "bin/imatrix" "${LOCAL_BIN_DIR}/llama-imatrix"; else echo "WARNING: bin/imatrix not found."; fi
    
    echo "INFO: llama.cpp executables copy attempt finished."
    touch "${LLAMA_CPP_COMPILED_MARKER}" 
else
    echo "INFO: llama.cpp already compiled with CUDA (marker found and tools likely present)."
fi
echo "-----------------------------------------------------"

# --- 3. Start the main application ---
echo "INFO: Compilation completed, you main now stop and commit the image."
cd "${APP_DIR}" 
exec :