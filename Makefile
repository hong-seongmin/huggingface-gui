# HuggingFace GUI Makefile
# Provides convenient commands for development and deployment

.PHONY: help install install-dev clean test lint format run run-desktop run-api docker-build docker-run docker-compose health check-deps upgrade backup restore

# Default Python command
PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
UV := $(shell which uv 2>/dev/null)

# Project directories
PROJECT_DIR := $(shell pwd)
VENV_DIR := $(PROJECT_DIR)/.venv
CACHE_DIR := /tmp/hf_model_cache
BACKUP_DIR := $(PROJECT_DIR)/backups

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
help: ## Show this help message
	@echo "🚀 HuggingFace GUI Development Commands"
	@echo "======================================"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo
	@echo "📋 Environment:"
ifdef UV
	@echo "   Package Manager: $(GREEN)uv$(NC) (fast)"
else
	@echo "   Package Manager: $(YELLOW)pip$(NC) (fallback)"
endif
	@echo "   Python: $(PYTHON)"
	@echo "   Project: $(PROJECT_DIR)"

# Installation targets
install: ## Install dependencies for production
	@echo "$(BLUE)Installing production dependencies...$(NC)"
ifdef UV
	@uv sync --no-dev
else
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt
endif
	@$(MAKE) setup-env
	@echo "$(GREEN)✅ Production installation complete$(NC)"

install-dev: ## Install dependencies for development
	@echo "$(BLUE)Installing development dependencies...$(NC)"
ifdef UV
	@uv sync
else
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt
	@$(PYTHON) -m pip install -r requirements-dev.txt
endif
	@$(MAKE) setup-env
	@echo "$(GREEN)✅ Development installation complete$(NC)"

setup-env: ## Setup environment files and directories
	@echo "$(BLUE)Setting up environment...$(NC)"
	@if [ ! -f .env ]; then \
		if [ -f .env.example ]; then \
			cp .env.example .env; \
			echo "$(GREEN)✅ Created .env from template$(NC)"; \
		else \
			echo "$(YELLOW)⚠️ .env.example not found$(NC)"; \
		fi; \
	fi
	@mkdir -p $(CACHE_DIR) logs $(BACKUP_DIR)
	@echo "$(GREEN)✅ Environment setup complete$(NC)"

# Development targets
run: ## Run Streamlit application
	@echo "$(BLUE)Starting Streamlit application...$(NC)"
ifdef UV
	@uv run streamlit run app.py
else
	@$(PYTHON) -m streamlit run app.py
endif

run-desktop: ## Run CustomTkinter desktop application
	@echo "$(BLUE)Starting desktop application...$(NC)"
ifdef UV
	@uv run python run.py
else
	@$(PYTHON) run.py
endif

run-api: ## Run FastAPI server only
	@echo "$(BLUE)Starting FastAPI server...$(NC)"
ifdef UV
	@uv run python -c "from fastapi_server import create_app; import uvicorn; uvicorn.run(create_app(), host='127.0.0.1', port=8000)"
else
	@$(PYTHON) -c "from fastapi_server import create_app; import uvicorn; uvicorn.run(create_app(), host='127.0.0.1', port=8000)"
endif

# Testing and Quality
test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
ifdef UV
	@uv run pytest tests/ -v
else
	@$(PYTHON) -m pytest tests/ -v
endif

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
ifdef UV
	@uv run flake8 --max-line-length=88 --extend-ignore=E203,W503 .
	@uv run mypy . --ignore-missing-imports
else
	@$(PYTHON) -m flake8 --max-line-length=88 --extend-ignore=E203,W503 .
	@$(PYTHON) -m mypy . --ignore-missing-imports
endif

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
ifdef UV
	@uv run black .
	@uv run isort .
else
	@$(PYTHON) -m black .
	@$(PYTHON) -m isort .
endif
	@echo "$(GREEN)✅ Code formatted$(NC)"

# Health checks
health: ## Check application health
	@echo "$(BLUE)Running health checks...$(NC)"
	@if [ -f scripts/health_check.py ]; then \
ifdef UV
		uv run python scripts/health_check.py; \
else
		$(PYTHON) scripts/health_check.py; \
endif
	else \
		echo "$(YELLOW)Health check script not found$(NC)"; \
		$(MAKE) check-deps; \
	fi

check-deps: ## Check if all dependencies are installed
	@echo "$(BLUE)Checking dependencies...$(NC)"
	@$(PYTHON) -c "import streamlit, transformers, torch, fastapi, uvicorn; print('$(GREEN)✅ All core dependencies available$(NC)')" || echo "$(RED)❌ Missing dependencies$(NC)"

# Docker targets
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t huggingface-gui:latest .
	@echo "$(GREEN)✅ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	@docker run -p 8501:8501 -p 8000:8000 \
		-v "$(PROJECT_DIR)/model_cache:/app/model_cache" \
		-v "$(PROJECT_DIR)/logs:/app/logs" \
		--name huggingface-gui \
		--rm -it huggingface-gui:latest

docker-compose: ## Run with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✅ Services started$(NC)"
	@echo "Access the application at: http://localhost:8501"

docker-stop: ## Stop docker-compose services
	@docker-compose down
	@echo "$(GREEN)✅ Services stopped$(NC)"

# Maintenance targets
clean: ## Clean cache and temporary files
	@echo "$(BLUE)Cleaning cache and temporary files...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ *.egg-info/ .coverage
	@rm -f app_debug.log *.log
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

upgrade: ## Upgrade all dependencies
	@echo "$(BLUE)Upgrading dependencies...$(NC)"
ifdef UV
	@uv sync --upgrade
else
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install --upgrade -r requirements.txt
endif
	@echo "$(GREEN)✅ Dependencies upgraded$(NC)"

# Backup and restore
backup: ## Backup configuration and state
	@echo "$(BLUE)Creating backup...$(NC)"
	@mkdir -p $(BACKUP_DIR)
	@tar -czf "$(BACKUP_DIR)/backup_$(shell date +%Y%m%d_%H%M%S).tar.gz" \
		.env app_state.json logs/ --exclude=logs/*.log 2>/dev/null || true
	@echo "$(GREEN)✅ Backup created in $(BACKUP_DIR)$(NC)"

restore: ## Restore from latest backup (interactive)
	@echo "$(BLUE)Available backups:$(NC)"
	@ls -la $(BACKUP_DIR)/*.tar.gz 2>/dev/null || echo "$(YELLOW)No backups found$(NC)"
	@echo "Enter backup filename to restore (or press Enter to cancel):"
	@read backup_file; \
	if [ -n "$$backup_file" ] && [ -f "$(BACKUP_DIR)/$$backup_file" ]; then \
		tar -xzf "$(BACKUP_DIR)/$$backup_file"; \
		echo "$(GREEN)✅ Restored from $$backup_file$(NC)"; \
	else \
		echo "$(YELLOW)Restore cancelled$(NC)"; \
	fi

# System info
info: ## Show system information
	@echo "$(BLUE)System Information:$(NC)"
	@echo "=================="
	@echo "OS: $(shell uname -s) $(shell uname -r)"
	@echo "Python: $(shell $(PYTHON) --version 2>&1)"
ifdef UV
	@echo "UV: $(shell uv --version 2>&1)"
endif
	@echo "Git: $(shell git --version 2>&1 || echo 'Not installed')"
	@echo "Docker: $(shell docker --version 2>&1 || echo 'Not installed')"
	@echo "Memory: $(shell free -h 2>/dev/null | grep '^Mem:' | awk '{print $$2}' || echo 'Unknown')"
	@echo "GPU: $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'Not available')"

# Development server with auto-reload
dev: ## Start development server with auto-reload
	@echo "$(BLUE)Starting development server...$(NC)"
ifdef UV
	@uv run streamlit run app.py --server.runOnSave=true
else
	@$(PYTHON) -m streamlit run app.py --server.runOnSave=true
endif

# Quick setup for new contributors
quick-start: ## Quick setup for new users
	@echo "$(GREEN)🚀 HuggingFace GUI Quick Start$(NC)"
	@echo "==============================="
	@$(MAKE) install
	@$(MAKE) health
	@echo
	@echo "$(GREEN)✅ Setup complete! Run 'make run' to start the application$(NC)"