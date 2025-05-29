#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "Executing startup.sh script..."
# This script is expected to be at /usr/local/bin/startup.sh inside the container
/usr/local/bin/startup.sh

# Check if startup.sh indicated an error (e.g., by exit code).
# The 'set -e' at the beginning of this script will cause it to exit
# if startup.sh returns a non-zero exit code.

echo "Startup script finished. Now starting combined_app.py..."
# Execute the main application, replacing the shell process.
# combined_app.py is expected to be in the WORKDIR /home/builder/app
exec python3 /home/builder/app/combined_app.py "$@"