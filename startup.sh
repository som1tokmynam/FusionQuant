#!/bin/bash
echo "--- Robust startup.sh v2.1 (Corrected) ---"

# Define a list of directories to search for NVIDIA drivers
# We include the standard system path and the common NVIDIA container paths.
SEARCH_DIRS=(
    "/usr/local/nvidia/lib64"
    "/usr/local/nvidia/lib"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/lib64"
)

# --- Function to find the latest driver file and create a symlink ---
# Arguments: $1 = library name (e.g., libcuda.so), $2 = symlink name (e.g., libcuda.so.1)
create_driver_symlink() {
    local lib_pattern="$1"
    local symlink_name="$2"
    local target_file=""
    local found_dir=""

    echo "Searching for target for ${symlink_name}..."

    # Loop through our search directories
    for dir in "${SEARCH_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # Find the newest version of the library file in the current directory
            candidate=$(find "${dir}" -maxdepth 1 -name "${lib_pattern}.[0-9]*" -type f | sort -V | tail -n 1)
            if [ -n "$candidate" ]; then
                target_file=$candidate
                found_dir=$dir
                echo "Found candidate: ${target_file}"
                break # Stop searching once we find a match
            fi
        fi
    done

    # If we found a file, create the symlink in a standard location
    if [ -n "$target_file" ]; then
        # The symlink MUST be created in a directory that is in the LD_LIBRARY_PATH
        # We will use /usr/local/nvidia/lib64 as it's guaranteed to be there.
        SYMLINK_DIR="/usr/local/nvidia/lib64"
        
        # Ensure the target directory for the symlink exists
        mkdir -p "${SYMLINK_DIR}"

        echo "Creating symlink: ln -sfn \"${target_file}\" \"${SYMLINK_DIR}/${symlink_name}\""
        ln -sfn "${target_file}" "${SYMLINK_DIR}/${symlink_name}"
        echo "Verifying link:"
        ls -l "${SYMLINK_DIR}/${symlink_name}"
    else
        echo "ERROR: Could not find any file matching '${lib_pattern}.[0-9]*' in search paths."
    fi
    echo "----------------------------"
}

# Create symlinks for the essential libraries
create_driver_symlink "libcuda.so" "libcuda.so.1"
create_driver_symlink "libnvidia-ml.so" "libnvidia-ml.so.1"

echo "Script finished."