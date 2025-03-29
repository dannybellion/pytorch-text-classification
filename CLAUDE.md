# CLAUDE.md - Text Classification Project

## Project Commands
- Install: `uv add`
- Type check: `mypy src tests`
- Run tests: `pytest tests`
- Run single test: `pytest tests/path/to/test.py::test_name -v`
- Train model: `python -m src.train`

## Code Style Guidelines
- Use Python 3.11+ features
- Using UV as a package manager
- Format according to flake8 
- Sort imports with isort
- Use type annotations for all functions and class attributes
- Follow PEP 8 naming conventions (snake_case for functions/variables, PascalCase for classes)
- Organize imports: stdlib first, then third-party, then local
- Error handling: use try/except only when necessary, prefer specific exceptions
- Docstrings: use Google style docstrings for all public functions
- Keep functions small and focused on a single responsibility
- Use pathlib for file operations instead of os.path