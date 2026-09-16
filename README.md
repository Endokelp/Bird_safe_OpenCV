# birdsafe

Installation coverage checker for visible bird-safe window markings.

Status: early development, no measured results yet.

## Setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

## Checks

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python tools/sloplint.py
```
