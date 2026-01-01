# ML in Production Examples

A repository demonstrating various approaches and patterns for productionizing machine learning models, using a rental prediction use case.

## Getting Started

### Prerequisites
- Python 3.10+ (project uses 3.12 by default)
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

```bash
# Install dependencies and create virtual environment
# This uses Python 3.12 as pinned in .python-version
uv sync

# Or specify a different Python version (3.10+)
uv sync --python 3.11

# Activate virtual environment
source .venv/bin/activate
```

**Note:** If you use a different Python version, you may need to resolve dependency conflicts. The project is tested with Python 3.12.
