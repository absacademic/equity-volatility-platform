# Installation

## Local Python environment

Python 3.11 or newer is required.

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install the package and development tools.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the checks.

```bash
ruff check .
ruff format --check .
pytest --cov=vol_platform --cov-report=term-missing
```

## Docker

Build the image from the repository root.

```bash
docker build -t equity-volatility-platform:0.6.0 .
```

Display the CLI help.

```bash
docker run --rm equity-volatility-platform:0.6.0 --help
```

Mount the repository when generated outputs should remain on the host.

```bash
docker run --rm -v "${PWD}:/app" equity-volatility-platform:0.6.0 synthetic-week6 --output-dir data/interim/week6-demo
```
