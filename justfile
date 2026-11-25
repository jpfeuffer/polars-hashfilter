# Development commands for polars_hashfilter

default:
    @just --list

# Create virtual environment with uv
venv:
    uv sync --group dev

# Install in development mode
dev: venv
    uv run maturin develop

# Install in release mode
release: venv
    uv run maturin develop --release

# Run tests
test: dev
    uv run pytest tests/ -v

# Format code
fmt:
    cargo fmt
    ruff format python/ tests/

# Lint code
lint:
    cargo clippy --all-features
    ruff check python/ tests/

# Type check
typecheck:
    uv run mypy python/ tests/

# Build wheel
build:
    uv run maturin build --release

# Clean build artifacts
clean:
    cargo clean
    rm -rf .venv target dist *.egg-info
