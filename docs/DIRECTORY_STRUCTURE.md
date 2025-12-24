# COHERENT Directory Structure

This document defines the canonical directory structure for the COHERENT project.

## Root Directory
- `coherent/`: Main package directory. The single source of truth for the codebase.
- `docs/`: Documentation.
- `tests/`: Automated tests.

## Package Structure (`coherent/`)
The `coherent` package is organized into the following submodules:

### `coherent/core`
The core runtime and reasoning engine.
- `orchestrator.py`: Main entry point for the reasoning loop.
- `reasoning/`: Agent logic, System 2 reasoning.
- `memory/`: Optical Holographic Memory (System 1).
- `modules/`: Specialized math modules (Calculus, Geometry, etc.).

### `coherent/apps`
End-user applications built on top of the core engine.
- `edu/`: Educational Math Solver (Streamlit).
- `pro/`: Professional/Enterprise Analysis tools.
- `demo/`: Demonstration scenarios.

### `coherent/tools`
Utility scripts and development tools.
- `lsp/`: Language Server Protocol implementation for the DSL.
- `notebooks/`: Jupyter notebook utilities.

## Legacy Directories (To Be Removed)
- `causalscript/`: Old package name. Content should be migrated to `coherent/` and this directory deleted.
