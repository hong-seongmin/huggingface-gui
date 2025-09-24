#!/bin/bash

# HuggingFace GUI Auto Setup Script
# Supports Linux and macOS
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_MIN_VERSION="3.9"
REQUIRED_MEMORY_GB=4

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# System detection
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Version comparison
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Check Python version
check_python() {
    log_info "Checking Python installation..."

    local python_cmd=""
    if command_exists python3; then
        python_cmd="python3"
    elif command_exists python; then
        python_cmd="python"
    else
        log_error "Python is not installed. Please install Python ${PYTHON_MIN_VERSION}+ first."
        exit 1
    fi

    local python_version
    python_version=$($python_cmd --version 2>&1 | cut -d' ' -f2)

    if version_ge "$python_version" "$PYTHON_MIN_VERSION"; then
        log_success "Python $python_version is compatible"
        echo "$python_cmd"
    else
        log_error "Python $python_version is too old. Minimum required: $PYTHON_MIN_VERSION"
        exit 1
    fi
}

# Check system memory
check_memory() {
    log_info "Checking system memory..."

    local os_type
    os_type=$(detect_os)
    local memory_gb=0

    if [[ "$os_type" == "linux" ]]; then
        memory_gb=$(free -g | awk 'NR==2{print $2}')
    elif [[ "$os_type" == "macos" ]]; then
        local memory_bytes
        memory_bytes=$(sysctl hw.memsize | awk '{print $2}')
        memory_gb=$((memory_bytes / 1024 / 1024 / 1024))
    fi

    if [[ $memory_gb -lt $REQUIRED_MEMORY_GB ]]; then
        log_warning "System has ${memory_gb}GB RAM. Minimum ${REQUIRED_MEMORY_GB}GB recommended."
        log_warning "Application may run slowly with large models."
    else
        log_success "System memory: ${memory_gb}GB (sufficient)"
    fi
}

# Check GPU availability
check_gpu() {
    log_info "Checking GPU availability..."

    if command_exists nvidia-smi; then
        local gpu_count
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        if [[ $gpu_count -gt 0 ]]; then
            log_success "NVIDIA GPU detected ($gpu_count GPU(s))"
            return 0
        fi
    fi

    if [[ $(detect_os) == "macos" ]] && [[ $(uname -m) == "arm64" ]]; then
        log_success "Apple Silicon detected (MPS support available)"
        return 0
    fi

    log_warning "No GPU acceleration detected. Using CPU only."
    return 1
}

# Install uv package manager
install_uv() {
    if command_exists uv; then
        log_success "uv is already installed"
        return 0
    fi

    log_info "Installing uv package manager..."

    if command_exists curl; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        log_error "curl is required to install uv. Please install curl first."
        exit 1
    fi

    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    if command_exists uv; then
        log_success "uv installed successfully"
    else
        log_error "Failed to install uv"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."

    local os_type
    os_type=$(detect_os)

    if [[ "$os_type" == "linux" ]]; then
        # Check if we have package manager permissions
        if command_exists apt-get; then
            if sudo -n true 2>/dev/null; then
                sudo apt-get update -qq
                sudo apt-get install -y git build-essential curl
            else
                log_warning "No sudo permissions. Please install: git build-essential curl"
            fi
        elif command_exists yum; then
            if sudo -n true 2>/dev/null; then
                sudo yum install -y git gcc gcc-c++ make curl
            else
                log_warning "No sudo permissions. Please install: git gcc gcc-c++ make curl"
            fi
        fi
    elif [[ "$os_type" == "macos" ]]; then
        # Check if Xcode command line tools are installed
        if ! xcode-select -p >/dev/null 2>&1; then
            log_info "Installing Xcode command line tools..."
            xcode-select --install
            log_warning "Please complete Xcode command line tools installation and re-run this script."
            exit 1
        fi
    fi
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."

    # Create .env file if it doesn't exist
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        if [[ -f "$SCRIPT_DIR/.env.example" ]]; then
            cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
            log_success "Created .env file from template"
        else
            log_warning ".env.example not found, creating basic .env file"
            cat > "$SCRIPT_DIR/.env" << 'EOF'
# HuggingFace GUI Configuration
HOST=127.0.0.1
PORT=8501
FASTAPI_HOST=127.0.0.1
FASTAPI_PORT=8000
HF_MODEL_CACHE_DIR=/tmp/hf_model_cache
DEFAULT_DEVICE=auto
MAX_CONCURRENT_MODELS=2
LOG_LEVEL=INFO
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EOF
        fi
    else
        log_success ".env file already exists"
    fi

    # Create necessary directories
    local cache_dir="/tmp/hf_model_cache"
    if [[ ! -d "$cache_dir" ]]; then
        mkdir -p "$cache_dir"
        log_success "Created model cache directory: $cache_dir"
    fi

    local logs_dir="$SCRIPT_DIR/logs"
    if [[ ! -d "$logs_dir" ]]; then
        mkdir -p "$logs_dir"
        log_success "Created logs directory"
    fi
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."

    cd "$SCRIPT_DIR"

    if [[ -f "pyproject.toml" ]] && command_exists uv; then
        log_info "Using uv to install dependencies..."
        uv sync
        log_success "Dependencies installed with uv"
    elif [[ -f "requirements.txt" ]]; then
        log_info "Using pip to install dependencies..."
        local python_cmd
        python_cmd=$(check_python)
        $python_cmd -m pip install --upgrade pip
        $python_cmd -m pip install -r requirements.txt
        log_success "Dependencies installed with pip"
    else
        log_error "No pyproject.toml or requirements.txt found"
        exit 1
    fi
}

# Test installation
test_installation() {
    log_info "Testing installation..."

    cd "$SCRIPT_DIR"

    if command_exists uv; then
        # Test with uv
        if uv run python -c "import streamlit, transformers, torch; print('All modules imported successfully')"; then
            log_success "Installation test passed"
        else
            log_error "Installation test failed"
            return 1
        fi
    else
        # Test with pip
        local python_cmd
        python_cmd=$(check_python)
        if $python_cmd -c "import streamlit, transformers, torch; print('All modules imported successfully')"; then
            log_success "Installation test passed"
        else
            log_error "Installation test failed"
            return 1
        fi
    fi
}

# Main installation flow
main() {
    echo "=================================================="
    echo "🚀 HuggingFace GUI Auto Setup"
    echo "=================================================="
    echo

    # System checks
    check_python
    check_memory
    check_gpu

    # Installation
    install_system_deps
    install_uv
    setup_environment
    install_dependencies

    # Verification
    if test_installation; then
        echo
        echo "=================================================="
        log_success "🎉 Setup completed successfully!"
        echo "=================================================="
        echo
        echo "📋 Next steps:"
        echo "   1. Review/edit .env file if needed"
        echo "   2. Run the application:"
        echo
        if command_exists uv; then
            echo "      # Streamlit version:"
            echo "      uv run streamlit run app.py"
            echo
            echo "      # Desktop version:"
            echo "      uv run python run.py"
        else
            echo "      # Streamlit version:"
            echo "      streamlit run app.py"
            echo
            echo "      # Desktop version:"
            echo "      python run.py"
        fi
        echo
        echo "   3. Open http://localhost:8501 in your browser"
        echo
        echo "💡 Use 'make help' for more commands"
        echo
    else
        log_error "Setup failed. Please check the error messages above."
        exit 1
    fi
}

# Run setup if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi