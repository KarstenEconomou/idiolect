set shell := ["sh", "-eu", "-c"]

# Run the complete test suite.
test:
    uv run pytest

# Check Python style and common errors.
lint:
    uv run ruff check .

# Type-check the source and tests.
typecheck:
    uv run ty check

# Run every required quality check.
check: lint typecheck test
