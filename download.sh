#!/bin/bash

# Function to check if a command is available
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Install gdown if not present
if ! command_exists gdown; then
  echo "Installing gdown..."
  pip install --upgrade --no-cache-dir gdown
fi

# Function to download file if it doesn't exist
download_file() {
  local file_id=$1
  local file_name=$2

  if [ ! -f "$file_name" ]; then
    echo "Downloading $file_name..."
    gdown "$file_id" -O "$file_name"
  else
    echo "$file_name already exists. Skipping download."
  fi
}

model_id= # Ask authors for ID
config_id= # Ask authors for ID

# Download files if they don't exist
download_file "$model_id" "model.keras"
download_file "$config_id" "config.json"
