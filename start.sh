#!/bin/bash

if [ ! -d "llama.cpp" ]; then
  # only run in dev env
  git clone https://github.com/ggerganov/llama.cpp
fi

export GGML_CUDA=OFF
if [[ -z "${RUN_LOCALLY}" ]]; then
  # enable CUDA if NOT running locally
  export GGML_CUDA=ON
fi

cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=${GGML_CUDA} -DLLAMA_CURL=OFF
cmake --build build --config Release -j --target llama-quantize llama-gguf-split llama-imatrix
cp ./build/bin/llama-* .
rm -rf build

cd ..
python combined_app.py
