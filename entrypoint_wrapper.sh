#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "Executing startup.sh script with sudo..."
# This is the fix: run the script with root privileges
sudo /usr/local/bin/startup.sh

# The 'set -e' at the beginning of this script will cause it to exit
# if startup.sh returns a non-zero exit code.

echo "Startup script finished. Now starting combined_app.py as user: $(whoami)"
# Execute the main application, replacing the shell process.
# This will run as the 'builder' user, as intended.
exec python3 /home/builder/app/combined_app.py "$@"