.PHONY: help setup start stop status logs test clean build-feast rebuild

help: ## Show this help message
	@echo "StrokePredict Spark-Feast Cluster"
	@echo "================================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Setup JARs and initial configuration
	@bash scripts/setup-jars.sh

build-feast: ## Build custom Feast Docker image
	@docker build -t custom-feast:latest ./feast/

start: ## Start the cluster
	@docker compose up -d
	@echo "Waiting for services to start..."
	@sleep 30
	@make status

stop: ## Stop the cluster
	@docker compose down

status: ## Check cluster status
	@echo "Service Status:"
	@docker compose ps
	@echo ""
	@echo "Health Checks:"
	@docker compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Health}}"

logs: ## View logs for all services
	@docker compose logs -f

test: ## Run cluster tests
	@echo "Testing Spark connection..."
	@python tests/test_spark_connection.py
	@echo "Testing Feast setup..."
	@python tests/test_feast_setup.py

clean: ## Clean up containers and volumes
	@docker compose down -v
	@docker system prune -f

rebuild: ## Rebuild and restart all services
	@make stop
	@make build-feast
	@make start

jupyter: ## Get Jupyter access token
	@docker logs stroke-jupyter | grep -o "http://[^ ]*token=[^ ]*"
