# Carcassonne Bot Battle
Welcome to SYNCS Bot Battle 2025! This year we will be using a modified rule set of the Carcassonne Board Game. 

## Getting started
This year we use Astral's [`uv`](https://docs.astral.sh/uv/) as our development environment.

### UV Route (Recommended)
> UV is a python version and package manager. It'll let you choose a python version and install as a portable folder.

To get started you will have to:
1. Install uv as seen in docs
2. Run `uv venv` in the current folder from your terminal (or IDE Interface)
    - This creates a python virtual environment managed by uv. The game engine and all packages should be run and installed using uv.
3. `source .venv/bin/activate` (for bash and zsh). Case for MacOS and Linux.
    - `source .venv/bin/activate.fish` (for fish)
    - `source .venv/bin/activate.bat` (for Windows Powershell)
4. Run `uv sync` in the current folder from your terminal (or IDE Interface)
    - Alternatively Run `uv pip install -e .`
    - You may need to run `deactivate` and then repeat just step 2
4. Use python as you normally would
    - On VS Code (and other IDEs) you will have configure the python interpreter to be the one we just created on uv

### Classic Route (At your own risk)

1. The game engine requires at least Python 3.12 (py312).
2. Create the environment `python -m venv .venv`
3. Activate the environment
4. Install requirements `pip install -e .`
