#!/bin/bash

# AEGIS Framework - Setup Script
# Advanced Encrypted Governance and Intelligence System
# This script sets up the AEGIS development environment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
AEGIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/..)" && pwd)"
REPO_URL="https://github.com/KaseMaster/Open-A.G.I.git"
PYTHON_VERSION="3.11"
NODE_VERSION="18"
DOCKER_COMPOSE_VERSION="2.20.0"

# Logging functions
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

log_header() {
    echo -e "${PURPLE}[AEGIS]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_system() {
    log_header "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        log_info "Detected Linux system"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_info "Detected macOS system"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" && "$ARCH" != "arm64" && "$ARCH" != "aarch64" ]]; then
        log_warning "Untested architecture: $ARCH"
    fi
    
    log_success "System check completed"
}

# Install system dependencies
install_system_deps() {
    log_header "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        if command_exists apt-get; then
            sudo apt-get update
            sudo apt-get install -y \
                curl \
                wget \
                git \
                build-essential \
                libssl-dev \
                libffi-dev \
                python3-dev \
                python3-pip \
                python3-venv \
                postgresql-client \
                redis-tools \
                tor \
                nginx \
                jq \
                htop \
                tree \
                unzip
        elif command_exists yum; then
            sudo yum update -y
            sudo yum install -y \
                curl \
                wget \
                git \
                gcc \
                gcc-c++ \
                make \
                openssl-devel \
                libffi-devel \
                python3-devel \
                python3-pip \
                postgresql \
                redis \
                tor \
                nginx \
                jq \
                htop \
                tree \
                unzip
        else
            log_error "Unsupported Linux distribution"
            exit 1
        fi
    elif [[ "$OS" == "macos" ]]; then
        if ! command_exists brew; then
            log_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        brew update
        brew install \
            curl \
            wget \
            git \
            python@3.11 \
            postgresql \
            redis \
            tor \
            nginx \
            jq \
            htop \
            tree
    fi
    
    log_success "System dependencies installed"
}

# Install Python and create virtual environment
setup_python() {
    log_header "Setting up Python environment..."
    
    # Check Python version
    if command_exists python3; then
        PYTHON_CURRENT=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        log_info "Current Python version: $PYTHON_CURRENT"
        
        if [[ "$PYTHON_CURRENT" < "$PYTHON_VERSION" ]]; then
            log_warning "Python $PYTHON_VERSION or higher is recommended"
        fi
    else
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Create virtual environment
    cd "$AEGIS_DIR"
    if [[ ! -d "venv" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing Python dependencies..."
        pip install -r requirements.txt
    fi
    
    if [[ -f "requirements-dev.txt" ]]; then
        log_info "Installing development dependencies..."
        pip install -r requirements-dev.txt
    fi
    
    log_success "Python environment setup completed"
}

# Install Node.js and npm dependencies
setup_nodejs() {
    log_header "Setting up Node.js environment..."
    
    # Check if Node.js is installed
    if ! command_exists node; then
        log_info "Installing Node.js..."
        
        if [[ "$OS" == "linux" ]]; then
            curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | sudo -E bash -
            sudo apt-get install -y nodejs
        elif [[ "$OS" == "macos" ]]; then
            brew install node@${NODE_VERSION}
        fi
    fi
    
    NODE_CURRENT=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    log_info "Current Node.js version: v$NODE_CURRENT"
    
    # Install global packages
    npm install -g \
        yarn \
        pm2 \
        nodemon \
        typescript \
        ts-node
    
    # Install project dependencies if package.json exists
    if [[ -f "package.json" ]]; then
        log_info "Installing Node.js dependencies..."
        npm install
    fi
    
    log_success "Node.js environment setup completed"
}

# Install Docker and Docker Compose
setup_docker() {
    log_header "Setting up Docker environment..."
    
    if ! command_exists docker; then
        log_info "Installing Docker..."
        
        if [[ "$OS" == "linux" ]]; then
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
        elif [[ "$OS" == "macos" ]]; then
            log_info "Please install Docker Desktop for macOS from https://docker.com/products/docker-desktop"
            log_warning "Manual installation required for macOS"
        fi
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose; then
        log_info "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    
    log_success "Docker environment setup completed"
}

# Generate SSL certificates
generate_certificates() {
    log_header "Generating SSL certificates..."
    
    CERT_DIR="$AEGIS_DIR/certs"
    mkdir -p "$CERT_DIR"
    
    if [[ ! -f "$CERT_DIR/server.crt" ]]; then
        log_info "Generating self-signed SSL certificate..."
        
        openssl req -x509 -newkey rsa:4096 -keyout "$CERT_DIR/server.key" -out "$CERT_DIR/server.crt" -days 365 -nodes -subj "/C=US/ST=State/L=City/O=AEGIS/OU=Security/CN=localhost"
        
        # Set proper permissions
        chmod 600 "$CERT_DIR/server.key"
        chmod 644 "$CERT_DIR/server.crt"
        
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

# Generate encryption keys
generate_keys() {
    log_header "Generating encryption keys..."
    
    KEY_DIR="$AEGIS_DIR/keys"
    mkdir -p "$KEY_DIR"
    
    if [[ ! -f "$KEY_DIR/master.key" ]]; then
        log_info "Generating master encryption key..."
        openssl rand -hex 32 > "$KEY_DIR/master.key"
        chmod 600 "$KEY_DIR/master.key"
    fi
    
    if [[ ! -f "$KEY_DIR/jwt.key" ]]; then
        log_info "Generating JWT secret key..."
        openssl rand -hex 64 > "$KEY_DIR/jwt.key"
        chmod 600 "$KEY_DIR/jwt.key"
    fi
    
    if [[ ! -f "$KEY_DIR/encryption.key" ]]; then
        log_info "Generating encryption key..."
        openssl rand -hex 32 > "$KEY_DIR/encryption.key"
        chmod 600 "$KEY_DIR/encryption.key"
    fi
    
    log_success "Encryption keys generated"
}

# Setup environment file
setup_environment() {
    log_header "Setting up environment configuration..."
    
    if [[ ! -f "$AEGIS_DIR/.env" ]]; then
        if [[ -f "$AEGIS_DIR/.env.example" ]]; then
            log_info "Creating .env file from template..."
            cp "$AEGIS_DIR/.env.example" "$AEGIS_DIR/.env"
            
            # Generate random passwords and keys
            JWT_SECRET=$(openssl rand -hex 64)
            POSTGRES_PASSWORD=$(openssl rand -hex 16)
            REDIS_PASSWORD=$(openssl rand -hex 16)
            ENCRYPTION_KEY=$(openssl rand -hex 32)
            MASTER_KEY=$(openssl rand -hex 32)
            SALT=$(openssl rand -hex 16)
            
            # Update .env file with generated values
            sed -i.bak "s/your-super-secret-jwt-key-change-this-immediately/$JWT_SECRET/g" "$AEGIS_DIR/.env"
            sed -i.bak "s/aegis_secure_pass_change_this/$POSTGRES_PASSWORD/g" "$AEGIS_DIR/.env"
            sed -i.bak "s/aegis_redis_pass_change_this/$REDIS_PASSWORD/g" "$AEGIS_DIR/.env"
            sed -i.bak "s/your-32-character-encryption-key/$ENCRYPTION_KEY/g" "$AEGIS_DIR/.env"
            sed -i.bak "s/your-master-key-for-key-derivation/$MASTER_KEY/g" "$AEGIS_DIR/.env"
            sed -i.bak "s/your-random-salt-value/$SALT/g" "$AEGIS_DIR/.env"
            
            # Remove backup file
            rm "$AEGIS_DIR/.env.bak"
            
            log_success "Environment file created with secure random values"
        else
            log_error ".env.example file not found"
            exit 1
        fi
    else
        log_info "Environment file already exists"
    fi
}

# Setup directories
setup_directories() {
    log_header "Creating directory structure..."
    
    mkdir -p "$AEGIS_DIR"/{data,logs,backups,certs,keys,temp}
    mkdir -p "$AEGIS_DIR/data"/{db,redis,prometheus,grafana}
    mkdir -p "$AEGIS_DIR/logs"/{app,nginx,tor}
    
    # Set proper permissions
    chmod 755 "$AEGIS_DIR"/{data,logs,backups,temp}
    chmod 700 "$AEGIS_DIR"/{certs,keys}
    
    log_success "Directory structure created"
}

# Install pre-commit hooks
setup_precommit() {
    log_header "Setting up pre-commit hooks..."
    
    if [[ -f "$AEGIS_DIR/.pre-commit-config.yaml" ]]; then
        source "$AEGIS_DIR/venv/bin/activate"
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    else
        log_warning "Pre-commit configuration not found"
    fi
}

# Run tests
run_tests() {
    log_header "Running initial tests..."
    
    cd "$AEGIS_DIR"
    source venv/bin/activate
    
    if command_exists pytest; then
        pytest tests/ -v --tb=short
        log_success "Tests completed"
    else
        log_warning "pytest not found, skipping tests"
    fi
}

# Display setup summary
show_summary() {
    log_header "Setup Summary"
    echo
    echo -e "${GREEN}✓${NC} System dependencies installed"
    echo -e "${GREEN}✓${NC} Python environment configured"
    echo -e "${GREEN}✓${NC} Node.js environment configured"
    echo -e "${GREEN}✓${NC} Docker environment configured"
    echo -e "${GREEN}✓${NC} SSL certificates generated"
    echo -e "${GREEN}✓${NC} Encryption keys generated"
    echo -e "${GREEN}✓${NC} Environment configuration created"
    echo -e "${GREEN}✓${NC} Directory structure created"
    echo -e "${GREEN}✓${NC} Pre-commit hooks installed"
    echo
    echo -e "${CYAN}Next Steps:${NC}"
    echo "1. Review and customize .env file"
    echo "2. Start services: make docker-up"
    echo "3. Run development server: make dev"
    echo "4. Access dashboard: http://localhost:8080"
    echo
    echo -e "${YELLOW}Important Security Notes:${NC}"
    echo "• Change default passwords in .env file"
    echo "• Review SSL certificate configuration"
    echo "• Configure firewall rules"
    echo "• Enable monitoring and logging"
    echo
    echo -e "${PURPLE}AEGIS Framework setup completed successfully!${NC}"
}

# Main setup function
main() {
    log_header "AEGIS Framework Setup"
    echo "Advanced Encrypted Governance and Intelligence System"
    echo "======================================================="
    echo
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        exit 1
    fi
    
    # Parse command line arguments
    SKIP_SYSTEM=false
    SKIP_DOCKER=false
    SKIP_TESTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-system)
                SKIP_SYSTEM=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-system    Skip system dependency installation"
                echo "  --skip-docker    Skip Docker installation"
                echo "  --skip-tests     Skip running tests"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run setup steps
    check_system
    
    if [[ "$SKIP_SYSTEM" != true ]]; then
        install_system_deps
    fi
    
    setup_python
    setup_nodejs
    
    if [[ "$SKIP_DOCKER" != true ]]; then
        setup_docker
    fi
    
    generate_certificates
    generate_keys
    setup_environment
    setup_directories
    setup_precommit
    
    if [[ "$SKIP_TESTS" != true ]]; then
        run_tests
    fi
    
    show_summary
}

# Run main function
main "$@"