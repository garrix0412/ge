.PHONY: install download-data train dashboard test clean help

# Default target
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
PYTHON     ?= python3
PIP        ?= pip
STREAMLIT  ?= streamlit
PYTEST     ?= pytest

SRC_DIR    := src
CONFIG_DIR := configs
DATA_DIR   := data
RESULTS_DIR := results

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install the project and all dependencies
	$(PIP) install -e ".[dev]"
	@echo "Installation complete."

download-data: ## Download historical OHLCV data from exchange
	$(PYTHON) -m src.data.downloader \
		--config $(CONFIG_DIR)/data_config.yaml
	@echo "Data download complete."

train: ## Train all models defined in model_config.yaml
	$(PYTHON) -m src.training.train \
		--data-config $(CONFIG_DIR)/data_config.yaml \
		--feature-config $(CONFIG_DIR)/feature_config.yaml \
		--model-config $(CONFIG_DIR)/model_config.yaml
	@echo "Training complete."

dashboard: ## Launch the Streamlit monitoring dashboard
	$(STREAMLIT) run $(SRC_DIR)/dashboard/app.py -- \
		--config $(CONFIG_DIR)/dashboard_config.yaml

test: ## Run the test suite with coverage
	$(PYTEST) tests/ \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-v
	@echo "Tests complete. HTML coverage report in htmlcov/."

clean: ## Remove generated artifacts (data, results, caches)
	rm -rf $(DATA_DIR)/raw $(DATA_DIR)/processed $(DATA_DIR)/sequences $(DATA_DIR)/scalers
	rm -rf $(RESULTS_DIR) checkpoints mlruns wandb
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."
