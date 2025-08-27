Quick Setup (no repo mutation)

1) Create/activate venv (only if needed)
   - Windows PowerShell:
     - Create: `python -m venv .venv`
     - Activate: `& .\.venv\Scripts\Activate.ps1`

2) Install deps
   - `python -m pip install --upgrade pip setuptools wheel`
   - `pip install -r requirements.txt`
   - If you don’t have CUDA: `pip install --upgrade torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu`

3) Install the package
   - `pip install -e .`

4) Quick smoke
   - `python -m cinderforge_e.validate.cli`

Notes
- The repo ignores `.venv/`, `__pycache__/`, `reports/`, and `results/` to keep history lean.
- To track a specific reports/results subfolder, add a negation rule in `.gitignore` (e.g., `!reports/public/`).
