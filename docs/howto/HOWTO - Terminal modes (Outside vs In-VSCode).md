# HOWTO — Know which terminal to use (OUTSIDE vs IN‑VSCODE)

**ABBRs used:** VS Code (Visual Studio Code), venv (virtual environment)

- **OUTSIDE** = external PowerShell 7 window. Use when you need clean environment or admin operations.  
- **IN‑VSCODE** = VS Code integrated terminal. Use when commands interact with the **open workspace**.

**Rule I’ll follow:** Every command block I give you will start with the mode label and set the working folder explicitly.

Example I will provide:
```
# [RUN: OUTSIDE]
Set-Location "C:\Users\<you>\dev\cinderforge_e"
.\.venv-gpu\Scripts\Activate.ps1
```