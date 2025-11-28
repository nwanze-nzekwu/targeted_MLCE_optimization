#!/usr/bin/env bash

# Artemis Tools Installer for version 3.1.1
# This script downloads and installs the Artemis Tools bundle

set -e # Exit on error

# Configuration
VERSION="3.1.1"
BASE_URL="https://files.artemis.turintech.ai/tools-bundle"
DEFAULT_USERNAME="Artemis_User"
DEFAULT_PASSWORD="Artemis_Custom_Runner_2025"
PYTHON_VERSION="3.11"

# Check for required commands
check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: Required command '$1' is not installed." >&2
        echo "Please install $2 and try again." >&2
        exit 1
    fi
}

# Check all required commands upfront
check_command "mktemp" "mktemp (part of coreutils)"

# Check for extraction tools - we need either unzip or tar
if ! command -v unzip >/dev/null 2>&1 && ! command -v tar >/dev/null 2>&1; then
    echo "Error: Neither 'unzip' nor 'tar' is installed." >&2
    echo "Please install either unzip or tar to extract files." >&2
    echo "  - On Ubuntu/Debian: sudo apt-get install unzip" >&2
    echo "  - On CentOS/RHEL: sudo yum install unzip" >&2
    echo "  - tar is usually pre-installed on most Unix systems" >&2
    exit 1
fi

# Check for either curl or wget
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    echo "Error: Neither 'curl' nor 'wget' is installed." >&2
    echo "Please install either curl or wget to download files." >&2
    echo "  - On Ubuntu/Debian: sudo apt-get install curl" >&2
    echo "  - On CentOS/RHEL: sudo yum install curl" >&2
    echo "  - On macOS: curl is pre-installed, or use 'brew install curl'" >&2
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
INSTALL_DIR=""
FORCE=false
SOURCE_BASHRC="${ARTEMIS_SOURCE_BASHRC:-true}"

# Logging function
log() {
    local level=$1
    shift
    local message="$@"

    case $level in
    ERROR)
        echo -e "${RED}[ERROR] $message${NC}" >&2
        ;;
    WARNING)
        echo -e "${YELLOW}[WARNING] $message${NC}"
        ;;
    INFO | *)
        echo -e "${GREEN}[INFO]${NC} $message"
        ;;
    esac
}

# Print usage
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --install-dir DIR    Installation directory (default: ./artemis-tools-$VERSION)
    --force             Force installation even if directory exists
    --no-source-bashrc  Do not source user's .bashrc in the activated shell
    -h, --help          Show this help message

Environment variables:
    ARTEMIS_DOWNLOAD_USERNAME    Username for download authentication
    ARTEMIS_DOWNLOAD_PASSWORD    Password for download authentication
    ARTEMIS_DOWNLOAD_TOOL        Force specific download tool: 'curl', 'wget', or 'auto' (default)
    ARTEMIS_DOWNLOAD_FORMAT      Force specific format: 'zip', 'tar', or 'auto' (default)
    ARTEMIS_SOURCE_BASHRC        Source user's .bashrc in activated shell: 'true' (default) or 'false'
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --install-dir)
        INSTALL_DIR="$2"
        shift 2
        ;;
    --force)
        FORCE=true
        shift
        ;;
    --no-source-bashrc)
        SOURCE_BASHRC=false
        shift
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        log ERROR "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
done

# Set default install directory if not specified
if [ -z "$INSTALL_DIR" ]; then
    INSTALL_DIR="./artemis-tools-$VERSION"
fi

# Convert to absolute path
INSTALL_DIR=$(cd "$(dirname "$INSTALL_DIR")" 2>/dev/null && pwd)/$(basename "$INSTALL_DIR")

# Get credentials from environment or use defaults
USERNAME="${ARTEMIS_DOWNLOAD_USERNAME:-$DEFAULT_USERNAME}"
PASSWORD="${ARTEMIS_DOWNLOAD_PASSWORD:-$DEFAULT_PASSWORD}"

# URLs
DOWNLOAD_URL="$BASE_URL/artemis-tools-$VERSION.zip"
LATEST_URL="$BASE_URL/artemis-tools-latest.zip"
DOWNLOAD_TAR_URL="$BASE_URL/artemis-tools-$VERSION.tar.gz"
LATEST_TAR_URL="$BASE_URL/artemis-tools-latest.tar.gz"

# Determine which format to use
USE_FORMAT="${ARTEMIS_DOWNLOAD_FORMAT:-auto}"

case "$USE_FORMAT" in
zip)
    # Force zip format
    if ! command -v unzip >/dev/null 2>&1; then
        log ERROR "zip format requested but unzip is not installed"
        exit 1
    fi
    ;;
tar | tar.gz | tgz)
    # Force tar.gz format
    if ! command -v tar >/dev/null 2>&1; then
        log ERROR "tar format requested but tar is not installed"
        exit 1
    fi
    USE_FORMAT="tar"
    DOWNLOAD_URL="$DOWNLOAD_TAR_URL"
    LATEST_URL="$LATEST_TAR_URL"
    ;;
auto | *)
    # Auto-detect based on available tools
    if command -v unzip >/dev/null 2>&1; then
        USE_FORMAT="zip"
    else
        USE_FORMAT="tar"
        DOWNLOAD_URL="$DOWNLOAD_TAR_URL"
        LATEST_URL="$LATEST_TAR_URL"
    fi
    ;;
esac

log INFO "Starting Artemis Tools $VERSION installation"
log INFO "Target directory: $INSTALL_DIR"

# Log format selection
if [ "${ARTEMIS_DOWNLOAD_FORMAT:-auto}" = "auto" ]; then
    if [ "$USE_FORMAT" = "tar" ]; then
        log INFO "Using tar.gz format (auto-detected: unzip not available)"
    else
        log INFO "Using zip format (auto-detected)"
    fi
else
    log INFO "Using $USE_FORMAT format (forced by ARTEMIS_DOWNLOAD_FORMAT)"
fi

# Check if installation directory exists
if [ -d "$INSTALL_DIR" ]; then
    log WARNING "Installation directory already exists"
    log WARNING ""
    log WARNING "Overwriting may fail if:"
    log WARNING "  - The virtual environment is currently activated"
    log WARNING "  - The artemis-runner is being used in another shell"
    log WARNING "  - Files are locked by another process"
    log WARNING ""
    log WARNING "If overwrite fails, please close all shells using this installation and try again."

    if [ "$FORCE" = true ]; then
        log INFO "Removing existing installation at $INSTALL_DIR"
        rm -rf "$INSTALL_DIR"
    else
        read -p "Installation directory $INSTALL_DIR already exists. Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log INFO "Installation cancelled"
            exit 0
        fi
        log INFO "Removing existing installation at $INSTALL_DIR"
        rm -rf "$INSTALL_DIR"
    fi
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d -t artemis-installer-XXXXXX)
trap "rm -rf $TEMP_DIR" EXIT

# Download function with authentication
download_file() {
    local url=$1
    local output=$2

    log INFO "Downloading from $url"

    # Check if ARTEMIS_DOWNLOAD_TOOL is set to force a specific tool
    local download_tool="${ARTEMIS_DOWNLOAD_TOOL:-auto}"

    case "$download_tool" in
    curl)
        if command -v curl >/dev/null 2>&1; then
            log INFO "Using curl (forced by ARTEMIS_DOWNLOAD_TOOL)"
            # Use digest authentication
            curl -f -L --digest -u "$USERNAME:$PASSWORD" -o "$output" "$url" 2>/dev/null || return 1
        else
            log ERROR "curl requested but not found. Please install curl."
            exit 1
        fi
        ;;
    wget)
        if command -v wget >/dev/null 2>&1; then
            log INFO "Using wget (forced by ARTEMIS_DOWNLOAD_TOOL)"
            # Use digest authentication with wget
            wget --auth-no-challenge --user="$USERNAME" --password="$PASSWORD" -O "$output" "$url" 2>/dev/null || return 1
        else
            log ERROR "wget requested but not found. Please install wget."
            exit 1
        fi
        ;;
    auto | *)
        # Auto-detect: prefer curl, fall back to wget
        if command -v curl >/dev/null 2>&1; then
            log INFO "Using curl (auto-detected)"
            # Use digest authentication
            curl -f -L --digest -u "$USERNAME:$PASSWORD" -o "$output" "$url" 2>/dev/null || return 1
        elif command -v wget >/dev/null 2>&1; then
            log INFO "Using wget (auto-detected)"
            # Use digest authentication with wget
            wget --auth-no-challenge --user="$USERNAME" --password="$PASSWORD" -O "$output" "$url" 2>/dev/null || return 1
        else
            log ERROR "Neither curl nor wget found. Please install one of them."
            exit 1
        fi
        ;;
    esac

    return 0
}

# Download the bundle
if [ "$USE_FORMAT" = "zip" ]; then
    BUNDLE_PATH="$TEMP_DIR/artemis-tools-$VERSION.zip"
else
    BUNDLE_PATH="$TEMP_DIR/artemis-tools-$VERSION.tar.gz"
fi

if ! download_file "$DOWNLOAD_URL" "$BUNDLE_PATH"; then
    log WARNING "Version-specific download failed, trying latest version"
    if ! download_file "$LATEST_URL" "$BUNDLE_PATH"; then
        log ERROR "Failed to download Artemis tools bundle"
        exit 1
    fi
fi

log INFO "Successfully downloaded bundle ($USE_FORMAT format)"

# Extract the bundle
log INFO "Extracting bundle..."
mkdir -p "$TEMP_DIR/extracted"

if [ "$USE_FORMAT" = "zip" ]; then
    unzip -q "$BUNDLE_PATH" -d "$TEMP_DIR/extracted"
else
    tar -xzf "$BUNDLE_PATH" -C "$TEMP_DIR/extracted"
fi

# Find the artemis-tools directory
BUNDLE_DIR="$TEMP_DIR/extracted/artemis-tools"
if [ ! -d "$BUNDLE_DIR" ]; then
    # Look for any directory containing wheels
    for dir in "$TEMP_DIR/extracted"/*; do
        if [ -d "$dir/wheels" ]; then
            BUNDLE_DIR="$dir"
            break
        fi
    done

    if [ ! -d "$BUNDLE_DIR/wheels" ]; then
        log ERROR "Could not find artemis-tools directory in extracted bundle"
        exit 1
    fi
fi

# Copy bundle to installation directory
log INFO "Installing Artemis tools to $INSTALL_DIR"
cp -r "$BUNDLE_DIR" "$INSTALL_DIR"

# Install uv
log INFO "Installing uv..."
UV_DIR="$INSTALL_DIR/.uv"
mkdir -p "$UV_DIR"

# Download and run uv installer
UV_INSTALLER="$UV_DIR/install.sh"
if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh -o "$UV_INSTALLER"
else
    wget -q https://astral.sh/uv/install.sh -O "$UV_INSTALLER"
fi

chmod +x "$UV_INSTALLER"
UV_INSTALL_DIR="$UV_DIR" sh "$UV_INSTALLER" >/dev/null 2>&1
rm -f "$UV_INSTALLER"

UV_CMD="$UV_DIR/uv"
if [ ! -x "$UV_CMD" ]; then
    log ERROR "uv installation failed"
    exit 1
fi

# Create Python 3.11 virtual environment
log INFO "Creating Python $PYTHON_VERSION virtual environment..."
cd "$INSTALL_DIR"
"$UV_CMD" venv --python "$PYTHON_VERSION" venv >/dev/null 2>&1 || {
    log ERROR "Failed to create Python $PYTHON_VERSION virtual environment"
    log ERROR "Please ensure Python $PYTHON_VERSION is available on your system"
    exit 1
}

# Install pip in the virtual environment
log INFO "Installing pip in virtual environment..."
"$UV_CMD" pip install --python "$INSTALL_DIR/venv/bin/python" pip >/dev/null 2>&1 || {
    log WARNING "Failed to install pip in virtual environment"
}

# Find artemis-runner wheel and extract version
WHEELS_DIR="$INSTALL_DIR/wheels"
if [ -d "$WHEELS_DIR" ]; then
    ARTEMIS_WHEEL=$(ls "$WHEELS_DIR"/artemis_runner-*.whl 2>/dev/null | head -n1)
    if [ -n "$ARTEMIS_WHEEL" ]; then
        # Extract version from wheel filename
        WHEEL_NAME=$(basename "$ARTEMIS_WHEEL" .whl)
        ARTEMIS_VERSION=$(echo "$WHEEL_NAME" | cut -d'-' -f2)

        log INFO "Installing artemis-runner==$ARTEMIS_VERSION in virtual environment..."
        "$UV_CMD" pip install --python "$INSTALL_DIR/venv/bin/python" \
            --prerelease=allow --find-links "$WHEELS_DIR" \
            "artemis-runner==$ARTEMIS_VERSION" >/dev/null 2>&1 || {
            log ERROR "Failed to install artemis-runner"
            log INFO "You can manually install using:"
            log INFO "  cd $INSTALL_DIR && .uv/uv pip install --python venv/bin/python --find-links wheels artemis-runner==$ARTEMIS_VERSION"
        }
    else
        log WARNING "No artemis-runner wheel found in wheels directory"
    fi
else
    log WARNING "No wheels directory found, skipping Python package installation"
fi

log INFO "Installation completed successfully!"
log INFO "Artemis tools installed to: $INSTALL_DIR"
log INFO ""

# Create activation script
cat >"$INSTALL_DIR/activate.sh" <<'ACTIVATE_SCRIPT'
#!/bin/bash
# Activate the artemis virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

echo "Virtual environment activated!"
echo ""
echo "To get started:"
echo "  1. Run: artemis-runner"
echo "  2. You'll be prompted for credentials on first run"
echo ""
echo "For help: artemis-runner --help"
ACTIVATE_SCRIPT

chmod +x "$INSTALL_DIR/activate.sh"

# Launch shell with activated environment
log INFO "Launching shell with activated virtual environment..."
log INFO ""
log INFO "To get started:"
log INFO "  1. Run: artemis-runner"
log INFO "  2. You'll be prompted for credentials on first run"
log INFO ""
log INFO "For help: artemis-runner --help"
log INFO ""

# Create a temporary bashrc file
TEMP_BASHRC="$INSTALL_DIR/.scripts/artemis_bashrc"
mkdir -p "$INSTALL_DIR/.scripts"

# Build the bashrc content based on whether we're sourcing user's bashrc
if [ "$SOURCE_BASHRC" = "false" ]; then
    cat >"$TEMP_BASHRC" <<EOF
# Activate the virtual environment
source "$INSTALL_DIR/venv/bin/activate"

# Set custom prompt since we're not loading user's bashrc
export PS1="\[\033[01;32m\]\u@\w\[\033[00m\]:\[\033[01;34m\](venv)\[\033[00m\]\$ "

# Change to installation directory
cd "$INSTALL_DIR"
EOF
else
    cat >"$TEMP_BASHRC" <<EOF
# Source user's bashrc first to get their environment
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Activate the virtual environment
source "$INSTALL_DIR/venv/bin/activate"

# Change to installation directory
cd "$INSTALL_DIR"
EOF
fi

# Check if we're running in a pipe
if [ -t 0 ] && [ -t 1 ]; then
    # Running interactively - launch new shell
    exec bash --rcfile "$TEMP_BASHRC"
else
    # Running in a pipe - just provide activation instructions
    log INFO ""
    log INFO "Installation complete! To activate the environment:"
    log INFO "  cd $INSTALL_DIR && source activate.sh"
    log INFO ""
    log INFO "Or start a new shell with the environment activated:"
    log INFO "  cd $INSTALL_DIR && bash --rcfile .scripts/artemis_bashrc"
fi
