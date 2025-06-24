#!binbash
set -e # Exit immediately if a command exits with a non-zero status.

echo =====================================================
echo ===== RUNTIME COMPILATION & SETUP SCRIPT v2.2 =====
echo =====================================================

# Environment variables are inherited from the Docker image
export PATH=${HOME_DIR}.localbin${PATH}
export PYTHONPATH=${HOME_DIR}.locallibpython3.10site-packages${PYTHONPATH}

# --- 1. Compile and Install ExLlamaV2 Extensions from Source ---
echo INFO Installing ExLlamaV2 from source to build CUDA extensions...
cd ${EXLLAMAV2_REPO_DIR}
pip install --user --no-cache-dir --verbose .

echo INFO Verifying ExLlamaV2 extension build...
python3 -c from exllamav2.ext import exllamav2_ext; print('SUCCESS ExLlamaV2 CUDA extension loaded successfully!', exllamav2_ext)
if [ $ -ne 0 ]; then
    echo ERROR Failed to compile or load ExLlamaV2 extensions.
    exit 1
fi
echo -----------------------------------------------------


# --- 2. Compile llama.cpp ---
echo INFO Compiling llama.cpp with CUDA support...
cd ${LLAMA_CPP_DIR}
rm -rf build
mkdir build && cd build

echo INFO Running CMake for llama.cpp...
# THE FIX Add LLAMA_BUILD_UTILS=ON to ensure tools like 'quantize' are built.
cmake .. -DLLAMA_CUDA=ON -DLLAMA_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF 
         -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON 
         -DLLAMA_BUILD_UTILS=ON -DLLAMA_SERVER=OFF

NPROC=$(nproc --ignore=1)
echo INFO Building llama.cpp with ${NPROC} cores...
cmake --build . --config Release -- -j${NPROC}

echo INFO Copying llama.cpp executables to ${HOME_DIR}.localbin...
# THE ROBUSTNESS IMPROVEMENT Check for file existence before copying.
if [ -f binquantize ]; then
    cp binquantize ${HOME_DIR}.localbinllama-quantize
    echo INFO Copied 'quantize' tool.
else
    echo WARNING 'quantize' executable not found in build artifacts. Skipping.
fi

if [ -f binmain ]; then
    cp binmain ${HOME_DIR}.localbinllama-cli
    echo INFO Copied 'main' (llama-cli) tool.
else
    echo WARNING 'main' executable not found in build artifacts. Skipping.
fi
echo -----------------------------------------------------


# --- 3. Finalization ---
echo 
echo =====================================================
echo       COMPILATION COMPLETE
echo =====================================================
echo You can now stop this container and use 'docker commit'
echo to save this state as your new pre-compiled base image.
echo The container will now sleep indefinitely. Press Ctrl+C to stop it.
echo =====================================================
echo 

sleep infinity