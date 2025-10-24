#!/bin/bash
# AEGIS Framework - Quick Start Script
# Automated setup and initialization for new users
# Version: 2.0.0

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.11"
VENV_DIR=".venv"

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║         🛡️  AEGIS FRAMEWORK QUICK START  🛡️              ║"
    echo "║                                                           ║"
    echo "║     Advanced Enterprise-Grade Intelligence System        ║"
    echo "║                   Version 2.0.0                           ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

# Check Python version
check_python() {
    echo -e "${CYAN}📋 Checking Python version...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is not installed${NC}"
        echo "Please install Python 3.11+ and try again"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}✅ Python ${PYTHON_VERSION} found${NC}"
    echo ""
}

# Create virtual environment
create_venv() {
    echo -e "${CYAN}🔧 Setting up virtual environment...${NC}"
    
    if [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            python3 -m venv "$VENV_DIR"
            echo -e "${GREEN}✅ Virtual environment recreated${NC}"
        else
            echo -e "${BLUE}ℹ️  Using existing virtual environment${NC}"
        fi
    else
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    fi
    echo ""
}

# Install dependencies
install_dependencies() {
    echo -e "${CYAN}📦 Installing dependencies...${NC}"
    echo "This may take a few minutes..."
    echo ""
    
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel --quiet
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --quiet
        echo -e "${GREEN}✅ Dependencies installed${NC}"
    else
        echo -e "${RED}❌ requirements.txt not found${NC}"
        exit 1
    fi
    echo ""
}

# Run tests
run_tests() {
    echo -e "${CYAN}🧪 Running tests...${NC}"
    
    source "$VENV_DIR/bin/activate"
    
    if python -m pytest tests/ -v --tb=short; then
        echo -e "${GREEN}✅ All tests passed!${NC}"
    else
        echo -e "${YELLOW}⚠️  Some tests failed, but you can continue${NC}"
    fi
    echo ""
}

# Run benchmarks
run_benchmarks() {
    echo -e "${CYAN}⚡ Running benchmarks (this will take ~1 minute)...${NC}"
    
    source "$VENV_DIR/bin/activate"
    
    if python benchmarks/benchmark_suite.py > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Benchmarks completed${NC}"
        
        if [ -f "benchmarks/benchmark_results.json" ]; then
            echo -e "${BLUE}📊 Results saved to: benchmarks/benchmark_results.json${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Benchmarks failed, but you can continue${NC}"
    fi
    echo ""
}

# Run integration example
run_example() {
    echo -e "${CYAN}🚀 Running integration example...${NC}"
    echo ""
    
    source "$VENV_DIR/bin/activate"
    
    python examples/06_complete_integration.py
    
    echo ""
    echo -e "${GREEN}✅ Integration example completed${NC}"
    echo ""
}

# Show usage instructions
show_usage() {
    echo -e "${PURPLE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                    🎉 SETUP COMPLETE! 🎉                  ║${NC}"
    echo -e "${PURPLE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}📚 Quick Start Commands:${NC}"
    echo ""
    echo -e "${YELLOW}1. Activate virtual environment:${NC}"
    echo "   source $VENV_DIR/bin/activate"
    echo ""
    echo -e "${YELLOW}2. Run health check:${NC}"
    echo "   python main.py health-check"
    echo ""
    echo -e "${YELLOW}3. Start AEGIS node (dry-run):${NC}"
    echo "   python main.py start-node --dry-run"
    echo ""
    echo -e "${YELLOW}4. Run tests:${NC}"
    echo "   pytest tests/ -v"
    echo ""
    echo -e "${YELLOW}5. Run benchmarks:${NC}"
    echo "   python benchmarks/benchmark_suite.py"
    echo ""
    echo -e "${YELLOW}6. View examples:${NC}"
    echo "   ls examples/"
    echo ""
    echo -e "${YELLOW}7. Read documentation:${NC}"
    echo "   cat docs/API_REFERENCE.md"
    echo ""
    echo -e "${CYAN}🐳 Docker Commands:${NC}"
    echo ""
    echo -e "${YELLOW}Build Docker image:${NC}"
    echo "   bash scripts/build_docker.sh"
    echo ""
    echo -e "${YELLOW}Start with Docker Compose:${NC}"
    echo "   docker-compose up -d"
    echo ""
    echo -e "${CYAN}📖 Documentation:${NC}"
    echo "   - README.md - Project overview"
    echo "   - docs/API_REFERENCE.md - Complete API documentation"
    echo "   - docs/ARCHITECTURE.md - System architecture"
    echo "   - docs/ROADMAP.md - Development roadmap"
    echo ""
    echo -e "${GREEN}✨ Happy coding with AEGIS Framework! ✨${NC}"
    echo ""
}

# Main execution
main() {
    print_banner
    
    # Check system requirements
    check_python
    
    # Setup environment
    create_venv
    install_dependencies
    
    # Ask user what to run
    echo -e "${CYAN}🎯 What would you like to do?${NC}"
    echo ""
    echo "1) Run tests"
    echo "2) Run benchmarks"
    echo "3) Run integration example"
    echo "4) All of the above"
    echo "5) Skip to usage instructions"
    echo ""
    read -p "Enter your choice (1-5): " -n 1 -r
    echo ""
    echo ""
    
    case $REPLY in
        1)
            run_tests
            ;;
        2)
            run_benchmarks
            ;;
        3)
            run_example
            ;;
        4)
            run_tests
            run_benchmarks
            run_example
            ;;
        5)
            echo -e "${BLUE}ℹ️  Skipping to usage instructions${NC}"
            echo ""
            ;;
        *)
            echo -e "${YELLOW}⚠️  Invalid choice, skipping to usage instructions${NC}"
            echo ""
            ;;
    esac
    
    # Show usage
    show_usage
}

# Run main function
main

exit 0
