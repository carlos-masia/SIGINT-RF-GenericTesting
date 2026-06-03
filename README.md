# SIGINT-RF-GenericTesting

Workspace for TDR (Time Domain Reflectometry) analysis from Touchstone .s1p files and optional SSBR GUI app.

## Requirements

- Python 3.14+
- Poetry (Python package manager)
- Git

## Git Configuration

Configure Git to track commits with your name and email:

```powershell
git config --global user.name "Carlos Masia"
git config --global user.email "carlos.masia@iceye.com"
```

Verify your configuration:
```powershell
git config --global --list
```

## Quick Start

### 1. Create Virtual Environment
```powershell
python -m venv .venv
```

### 2. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Poetry
```powershell
pip install poetry
```

### 4. Install Project Dependencies
```powershell
poetry install
```

### 5. Activate Poetry Environment (Optional)
```powershell
poetry env activate
```

## Project Structure

```
SIGINT-RF-GenericTesting/
├── TDR/                      # TDR analysis scripts
│   ├── tdr_from_s1p.py      # Main TDR conversion script
│   └── kaiser_filter/        # Kaiser filter implementation
├── ssbr_app/                 # SSBR GUI application (optional)
│   └── cyphal_types/         # Generated CYPHAL type definitions
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Running Scripts

### TDR Script
```powershell
.\.venv\Scripts\python .\TDR\tdr_from_s1p.py
```

Or with Poetry:
```powershell
poetry run python .\TDR\tdr_from_s1p.py
```

## Dependencies

### Core Dependencies
- **numpy** (>=2.4.6) - Numerical computing
- **scipy** (>=1.10) - Scientific computing
- **matplotlib** (>=3.10.9) - Plotting and visualization
- **scikit-rf** (>=0.28) - RF network analysis

### Optional Dependencies

#### SSBR GUI App
```powershell
poetry install --extras ssbr
```
- **pyserial** (>=3.5) - Serial communication

#### Development Tools
```powershell
poetry install --extras dev
```
- **pytest** - Testing framework
- **pytest-cov** - Code coverage
- **black** - Code formatter
- **ruff** - Python linter
- **mypy** - Static type checker

## Development

### Run Tests
```powershell
poetry run pytest
```

### Format Code
```powershell
poetry run black .
```

### Lint Code
```powershell
poetry run ruff check .
```

### Type Checking
```powershell
poetry run mypy .
```

## Troubleshooting

### Virtual environment not activating
If `.\.venv\Scripts\Activate.ps1` fails, try:
```powershell
.\.venv\Scripts\activate.bat  # For Command Prompt
```

### Poetry not found
Ensure Poetry is installed globally:
```powershell
pip install --upgrade poetry
```

### Dependencies not installing
If `poetry install` fails, try installing without the project root:
```powershell
poetry install --no-root
```

## License

MIT

## Author

Enrico Barone
