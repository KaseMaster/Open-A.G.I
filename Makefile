# AEGIS Framework - Makefile
# Advanced Encrypted Governance and Intelligence System
# Automation for development, testing, and deployment tasks

.PHONY: help install install-dev test test-cov test-security test-performance lint format type-check security-scan clean build docs serve-docs docker docker-build docker-run docker-compose-up docker-compose-down backup monitor benchmark profile deploy-local deploy-docker deploy-k8s pre-commit setup-dev setup-prod validate-config check-deps update-deps release

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
DOCKER := docker
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl
PROJECT_NAME := aegis-framework
VERSION := $(shell python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
DOCKER_REGISTRY := ghcr.io/aegis-project
NAMESPACE := aegis-system

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(CYAN)AEGIS Framework - Development Automation$(NC)"
	@echo "$(BLUE)=========================================$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(PURPLE)Examples:$(NC)"
	@echo "  make install-dev    # Install development dependencies"
	@echo "  make test-cov       # Run tests with coverage"
	@echo "  make docker-build   # Build Docker image"
	@echo "  make deploy-local   # Deploy locally"

# Installation targets
install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install -e ".[dev,test,docs,monitoring,security,performance]"
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

install-all: ## Install all dependencies including cloud providers
	@echo "$(GREEN)Installing all dependencies...$(NC)"
	$(PIP) install -e ".[all]"

# Testing targets
test: ## Run basic tests
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ -v --cov=. --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "$(CYAN)Coverage report generated in htmlcov/$(NC)"

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) tests/ -v -m "unit"

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTEST) tests/ -v -m "integration"

test-security: ## Run security tests
	@echo "$(GREEN)Running security tests...$(NC)"
	$(PYTEST) tests/ -v -m "security"
	bandit -r . -f json -o security-report.json
	safety check --json --output safety-report.json

test-performance: ## Run performance tests and benchmarks
	@echo "$(GREEN)Running performance tests...$(NC)"
	$(PYTEST) tests/ -v -m "performance" --benchmark-only
	$(PYTHON) benchmarks/run_benchmarks.py

test-all: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	$(PYTEST) tests/ -v --cov=. --cov-report=term-missing

# Code quality targets
lint: ## Run linting checks
	@echo "$(GREEN)Running linting checks...$(NC)"
	ruff check .
	flake8 .
	mypy .

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	black .
	isort .
	ruff check . --fix

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checks...$(NC)"
	mypy .

security-scan: ## Run security scans
	@echo "$(GREEN)Running security scans...$(NC)"
	bandit -r . -ll
	safety check
	semgrep --config=auto .

quality-check: lint type-check security-scan ## Run all quality checks

# Documentation targets
docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	cd docs && sphinx-build -b html . _build/html
	@echo "$(CYAN)Documentation built in docs/_build/html/$(NC)"

serve-docs: docs ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	@echo "$(GREEN)Cleaning documentation build...$(NC)"
	rm -rf docs/_build/

# Build targets
clean: ## Clean build artifacts and cache
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf security-report.json
	rm -rf safety-report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: clean ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(NC)"
	$(PYTHON) -m build
	@echo "$(CYAN)Packages built in dist/$(NC)"

# Docker targets
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image: $(DOCKER_IMAGE)$(NC)"
	$(DOCKER) build -t $(DOCKER_IMAGE) .
	$(DOCKER) tag $(DOCKER_IMAGE) $(PROJECT_NAME):latest

docker-build-dev: ## Build development Docker image
	@echo "$(GREEN)Building development Docker image...$(NC)"
	$(DOCKER) build -f Dockerfile.dev -t $(PROJECT_NAME):dev .

docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(NC)"
	$(DOCKER) run -it --rm -p 8080:8080 -p 8181:8181 $(DOCKER_IMAGE)

docker-run-dev: ## Run development Docker container
	@echo "$(GREEN)Running development Docker container...$(NC)"
	$(DOCKER) run -it --rm -v $(PWD):/app -p 8080:8080 -p 8181:8181 $(PROJECT_NAME):dev

docker-push: ## Push Docker image to registry
	@echo "$(GREEN)Pushing Docker image to registry...$(NC)"
	$(DOCKER) tag $(DOCKER_IMAGE) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	$(DOCKER) push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)

docker-compose-up: ## Start services with docker-compose
	@echo "$(GREEN)Starting services with docker-compose...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(CYAN)Services started. Check status with: docker-compose ps$(NC)"

docker-compose-down: ## Stop services with docker-compose
	@echo "$(GREEN)Stopping services with docker-compose...$(NC)"
	$(DOCKER_COMPOSE) down

docker-compose-logs: ## Show docker-compose logs
	@echo "$(GREEN)Showing docker-compose logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

docker-clean: ## Clean Docker images and containers
	@echo "$(GREEN)Cleaning Docker images and containers...$(NC)"
	$(DOCKER) system prune -f
	$(DOCKER) image prune -f

# Development targets
setup-dev: install-dev ## Setup development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	cp config/config.example.yml config/config.yml
	mkdir -p logs data backups certs
	@echo "$(YELLOW)Please edit config/config.yml with your settings$(NC)"
	@echo "$(GREEN)Development environment setup complete!$(NC)"

setup-prod: ## Setup production environment
	@echo "$(GREEN)Setting up production environment...$(NC)"
	mkdir -p logs data backups certs
	@echo "$(YELLOW)Please configure production settings in config/config.yml$(NC)"
	@echo "$(RED)Remember to set secure passwords and certificates!$(NC)"

validate-config: ## Validate configuration files
	@echo "$(GREEN)Validating configuration...$(NC)"
	$(PYTHON) -c "import yaml; yaml.safe_load(open('config/config.yml'))"
	@echo "$(GREEN)Configuration is valid!$(NC)"

# Dependency management
check-deps: ## Check for dependency updates
	@echo "$(GREEN)Checking for dependency updates...$(NC)"
	pip list --outdated

update-deps: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	pip install --upgrade pip
	pip install --upgrade -e ".[dev,test,docs,monitoring,security,performance]"

freeze-deps: ## Freeze current dependencies
	@echo "$(GREEN)Freezing dependencies...$(NC)"
	pip freeze > requirements-frozen.txt

# Monitoring and operations
monitor: ## Start monitoring dashboard
	@echo "$(GREEN)Starting monitoring dashboard...$(NC)"
	$(PYTHON) monitoring_dashboard.py

backup: ## Run backup system
	@echo "$(GREEN)Running backup system...$(NC)"
	$(PYTHON) backup_system.py --backup-all

restore: ## Restore from backup (requires BACKUP_FILE variable)
	@echo "$(GREEN)Restoring from backup: $(BACKUP_FILE)$(NC)"
	$(PYTHON) backup_system.py --restore $(BACKUP_FILE)

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(NC)"
	$(PYTHON) benchmarks/run_benchmarks.py --output benchmarks/results/

profile: ## Run performance profiling
	@echo "$(GREEN)Running performance profiling...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats main.py
	$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Deployment targets
deploy-local: setup-dev ## Deploy locally for development
	@echo "$(GREEN)Deploying locally...$(NC)"
	$(PYTHON) main.py --config config/config.yml

deploy-docker: docker-build docker-compose-up ## Deploy with Docker
	@echo "$(GREEN)Deployed with Docker!$(NC)"
	@echo "$(CYAN)Access the dashboard at: http://localhost:8080$(NC)"

deploy-k8s: ## Deploy to Kubernetes
	@echo "$(GREEN)Deploying to Kubernetes...$(NC)"
	$(KUBECTL) create namespace $(NAMESPACE) --dry-run=client -o yaml | $(KUBECTL) apply -f -
	$(KUBECTL) apply -f k8s/ -n $(NAMESPACE)
	@echo "$(CYAN)Deployed to Kubernetes namespace: $(NAMESPACE)$(NC)"

undeploy-k8s: ## Remove from Kubernetes
	@echo "$(GREEN)Removing from Kubernetes...$(NC)"
	$(KUBECTL) delete -f k8s/ -n $(NAMESPACE)

k8s-status: ## Check Kubernetes deployment status
	@echo "$(GREEN)Checking Kubernetes status...$(NC)"
	$(KUBECTL) get all -n $(NAMESPACE)

k8s-logs: ## Show Kubernetes logs
	@echo "$(GREEN)Showing Kubernetes logs...$(NC)"
	$(KUBECTL) logs -f deployment/aegis-node -n $(NAMESPACE)

# Security targets
generate-certs: ## Generate SSL certificates for development
	@echo "$(GREEN)Generating SSL certificates...$(NC)"
	mkdir -p certs
	openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=AEGIS/CN=localhost"
	@echo "$(CYAN)Certificates generated in certs/$(NC)"

generate-keys: ## Generate cryptographic keys
	@echo "$(GREEN)Generating cryptographic keys...$(NC)"
	$(PYTHON) -c "from crypto_framework import CryptoFramework; cf = CryptoFramework(); cf.generate_keypair(); print('Keys generated successfully')"

rotate-keys: ## Rotate cryptographic keys
	@echo "$(GREEN)Rotating cryptographic keys...$(NC)"
	$(PYTHON) -c "from crypto_framework import CryptoFramework; cf = CryptoFramework(); cf.rotate_keys(); print('Keys rotated successfully')"

# Database targets
db-init: ## Initialize database
	@echo "$(GREEN)Initializing database...$(NC)"
	$(PYTHON) -c "from storage_system import StorageSystem; ss = StorageSystem(); ss.initialize_database()"

db-migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	alembic upgrade head

db-backup: ## Backup database
	@echo "$(GREEN)Backing up database...$(NC)"
	$(PYTHON) backup_system.py --backup-database

db-restore: ## Restore database (requires DB_BACKUP_FILE variable)
	@echo "$(GREEN)Restoring database from: $(DB_BACKUP_FILE)$(NC)"
	$(PYTHON) backup_system.py --restore-database $(DB_BACKUP_FILE)

# Network targets
network-test: ## Test P2P network connectivity
	@echo "$(GREEN)Testing P2P network connectivity...$(NC)"
	$(PYTHON) -c "from p2p_network import P2PNetwork; p2p = P2PNetwork(); p2p.test_connectivity()"

network-status: ## Show network status
	@echo "$(GREEN)Showing network status...$(NC)"
	$(PYTHON) -c "from p2p_network import P2PNetwork; p2p = P2PNetwork(); p2p.show_status()"

# Consensus targets
consensus-test: ## Test consensus algorithm
	@echo "$(GREEN)Testing consensus algorithm...$(NC)"
	$(PYTHON) -c "from consensus_algorithm import ConsensusAlgorithm; ca = ConsensusAlgorithm(); ca.test_consensus()"

consensus-benchmark: ## Benchmark consensus performance
	@echo "$(GREEN)Benchmarking consensus performance...$(NC)"
	$(PYTHON) benchmarks/consensus_benchmark.py

# Pre-commit and CI targets
pre-commit: ## Run pre-commit hooks
	@echo "$(GREEN)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	@echo "$(GREEN)Updating pre-commit hooks...$(NC)"
	pre-commit autoupdate

ci-test: ## Run CI test suite
	@echo "$(GREEN)Running CI test suite...$(NC)"
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-scan
	$(MAKE) test-cov
	$(MAKE) test-security
	@echo "$(GREEN)CI test suite completed successfully!$(NC)"

# Release targets
release-check: ## Check if ready for release
	@echo "$(GREEN)Checking release readiness...$(NC)"
	$(MAKE) ci-test
	$(MAKE) build
	@echo "$(GREEN)Ready for release!$(NC)"

release-patch: ## Release patch version
	@echo "$(GREEN)Releasing patch version...$(NC)"
	cz bump --increment PATCH
	$(MAKE) build

release-minor: ## Release minor version
	@echo "$(GREEN)Releasing minor version...$(NC)"
	cz bump --increment MINOR
	$(MAKE) build

release-major: ## Release major version
	@echo "$(GREEN)Releasing major version...$(NC)"
	cz bump --increment MAJOR
	$(MAKE) build

# Utility targets
logs: ## Show application logs
	@echo "$(GREEN)Showing application logs...$(NC)"
	tail -f logs/aegis.log

logs-error: ## Show error logs only
	@echo "$(GREEN)Showing error logs...$(NC)"
	grep -i error logs/aegis.log | tail -20

status: ## Show system status
	@echo "$(GREEN)AEGIS System Status$(NC)"
	@echo "$(BLUE)==================$(NC)"
	@echo "$(YELLOW)Version:$(NC) $(VERSION)"
	@echo "$(YELLOW)Python:$(NC) $(shell python --version)"
	@echo "$(YELLOW)Docker:$(NC) $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "$(YELLOW)Kubernetes:$(NC) $(shell kubectl version --client --short 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "$(GREEN)Services Status:$(NC)"
	@ps aux | grep -E "(aegis|python)" | grep -v grep || echo "No AEGIS processes running"

info: ## Show project information
	@echo "$(CYAN)AEGIS Framework Information$(NC)"
	@echo "$(BLUE)============================$(NC)"
	@echo "$(YELLOW)Project:$(NC) $(PROJECT_NAME)"
	@echo "$(YELLOW)Version:$(NC) $(VERSION)"
	@echo "$(YELLOW)Docker Image:$(NC) $(DOCKER_IMAGE)"
	@echo "$(YELLOW)Registry:$(NC) $(DOCKER_REGISTRY)"
	@echo "$(YELLOW)Namespace:$(NC) $(NAMESPACE)"
	@echo ""
	@echo "$(GREEN)Key Features:$(NC)"
	@echo "  • Quantum-resistant cryptography"
	@echo "  • Decentralized P2P network"
	@echo "  • Hybrid consensus algorithm"
	@echo "  • Distributed storage system"
	@echo "  • Real-time monitoring"
	@echo "  • Automated backup system"

# Development workflow shortcuts
dev: install-dev setup-dev ## Quick development setup
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(CYAN)Run 'make deploy-local' to start the system$(NC)"

quick-test: ## Quick test run (unit tests only)
	@echo "$(GREEN)Running quick tests...$(NC)"
	$(PYTEST) tests/ -v -m "unit" --tb=short

full-check: clean install-dev ci-test ## Full quality check
	@echo "$(GREEN)Full quality check completed!$(NC)"

# Emergency targets
emergency-stop: ## Emergency stop all services
	@echo "$(RED)EMERGENCY STOP - Stopping all AEGIS services...$(NC)"
	pkill -f "python.*aegis" || true
	$(DOCKER_COMPOSE) down || true
	$(KUBECTL) scale deployment --replicas=0 --all -n $(NAMESPACE) || true
	@echo "$(RED)All services stopped!$(NC)"

emergency-backup: ## Emergency backup of critical data
	@echo "$(RED)EMERGENCY BACKUP - Backing up critical data...$(NC)"
	$(PYTHON) backup_system.py --emergency-backup
	@echo "$(RED)Emergency backup completed!$(NC)"

# System maintenance
maintenance-start: ## Start maintenance mode
	@echo "$(YELLOW)Starting maintenance mode...$(NC)"
	touch .maintenance
	@echo "$(YELLOW)System in maintenance mode$(NC)"

maintenance-stop: ## Stop maintenance mode
	@echo "$(GREEN)Stopping maintenance mode...$(NC)"
	rm -f .maintenance
	@echo "$(GREEN)System operational$(NC)"

health-check: ## Perform system health check
	@echo "$(GREEN)Performing system health check...$(NC)"
	$(PYTHON) -c "from monitoring_dashboard import HealthChecker; hc = HealthChecker(); hc.full_health_check()"

# Performance optimization
optimize: ## Run system optimization
	@echo "$(GREEN)Running system optimization...$(NC)"
	$(PYTHON) -c "from performance_optimizer import PerformanceOptimizer; po = PerformanceOptimizer(); po.optimize_system()"

cache-clear: ## Clear all caches
	@echo "$(GREEN)Clearing all caches...$(NC)"
	$(PYTHON) -c "from storage_system import StorageSystem; ss = StorageSystem(); ss.clear_cache()"

# Debugging targets
debug: ## Start in debug mode
	@echo "$(GREEN)Starting in debug mode...$(NC)"
	$(PYTHON) main.py --debug --verbose

debug-network: ## Debug network issues
	@echo "$(GREEN)Debugging network issues...$(NC)"
	$(PYTHON) debug_tools/network_debugger.py

debug-consensus: ## Debug consensus issues
	@echo "$(GREEN)Debugging consensus issues...$(NC)"
	$(PYTHON) debug_tools/consensus_debugger.py

debug-storage: ## Debug storage issues
	@echo "$(GREEN)Debugging storage issues...$(NC)"
	$(PYTHON) debug_tools/storage_debugger.py

# Integration with external tools
grafana-setup: ## Setup Grafana dashboards
	@echo "$(GREEN)Setting up Grafana dashboards...$(NC)"
	$(PYTHON) monitoring/setup_grafana.py

prometheus-config: ## Generate Prometheus configuration
	@echo "$(GREEN)Generating Prometheus configuration...$(NC)"
	$(PYTHON) monitoring/generate_prometheus_config.py

elk-setup: ## Setup ELK stack for logging
	@echo "$(GREEN)Setting up ELK stack...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.elk.yml up -d

# Custom targets for specific environments
dev-reset: ## Reset development environment
	@echo "$(YELLOW)Resetting development environment...$(NC)"
	$(MAKE) clean
	$(MAKE) docker-compose-down
	rm -rf data/* logs/* backups/*
	$(MAKE) setup-dev
	@echo "$(GREEN)Development environment reset!$(NC)"

prod-deploy: ## Production deployment checklist
	@echo "$(RED)PRODUCTION DEPLOYMENT CHECKLIST$(NC)"
	@echo "$(YELLOW)================================$(NC)"
	@echo "[ ] Configuration reviewed and secured"
	@echo "[ ] SSL certificates installed"
	@echo "[ ] Database backups verified"
	@echo "[ ] Monitoring configured"
	@echo "[ ] Security scans passed"
	@echo "[ ] Performance tests completed"
	@echo "[ ] Rollback plan prepared"
	@echo ""
	@echo "$(RED)Run 'make release-check' before deployment!$(NC)"

# Version information
version: ## Show version information
	@echo "$(CYAN)AEGIS Framework v$(VERSION)$(NC)"
	@echo "$(GREEN)Advanced Encrypted Governance and Intelligence System$(NC)"