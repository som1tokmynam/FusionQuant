#!/bin/bash

# Define the library directory (this might need to be more flexible)
LIB_DIR="/usr/lib/x86_64-linux-gnu"
# Or, attempt to find it, e.g., by looking where nvidia-smi is, then navigating,
# or checking common paths. This example assumes a known directory.

if [ ! -d "$LIB_DIR" ]; then
  echo "Error: Library directory $LIB_DIR not found."
  exit 1
fi

cd "$LIB_DIR" || exit

echo "Current directory: $(pwd)"

# --- Handle libcuda.so.1 ---
# Find candidate libcuda.so.X.Y.Z files (actual files, not symlinks)
# The pattern [0-9] helps ensure we match version numbers.
cuda_targets=$(find . -maxdepth 1 -name 'libcuda.so.[0-9]*' -type f -print0 | xargs -0 ls -1 2>/dev/null)

if [ -n "$cuda_targets" ]; then
  # Select the "latest" version using version sort
  latest_cuda_target=$(echo "$cuda_targets" | sort -V | tail -n 1)

  if [ -n "$latest_cuda_target" ] && [ -f "$latest_cuda_target" ]; then
    echo "Found target for libcuda.so.1: $latest_cuda_target"
    echo "Creating symlink: ln -sfn \"$latest_cuda_target\" libcuda.so.1"
    ln -sfn "$latest_cuda_target" libcuda.so.1
    echo "Verifying link:"
    ls -l libcuda.so.1
  else
    echo "Warning: No suitable concrete libcuda.so.X.Y.Z file found in $LIB_DIR."
  fi
else
  echo "Warning: No libcuda.so.X.Y.Z files found in $LIB_DIR."
fi

echo # Spacer

# --- Handle libnvidia-ml.so.1 ---
# Find candidate libnvidia-ml.so.X.Y.Z files
nvml_targets=$(find . -maxdepth 1 -name 'libnvidia-ml.so.[0-9]*' -type f -print0 | xargs -0 ls -1 2>/dev/null)

if [ -n "$nvml_targets" ]; then
  latest_nvml_target=$(echo "$nvml_targets" | sort -V | tail -n 1)

  if [ -n "$latest_nvml_target" ] && [ -f "$latest_nvml_target" ]; then
    echo "Found target for libnvidia-ml.so.1: $latest_nvml_target"
    echo "Creating symlink: ln -sfn \"$latest_nvml_target\" libnvidia-ml.so.1"
    ln -sfn "$latest_nvml_target" libnvidia-ml.so.1
    echo "Verifying link:"
    ls -l libnvidia-ml.so.1
  else
    echo "Warning: No suitable concrete libnvidia-ml.so.X.Y.Z file found in $LIB_DIR."
  fi
else
  echo "Warning: No libnvidia-ml.so.X.Y.Z files found in $LIB_DIR."
fi

echo
echo "Script finished."
echo "Modifying system library links can have unintended consequences. Proceed with caution."