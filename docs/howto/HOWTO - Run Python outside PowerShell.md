# HOWTO — Run Python outside PowerShell (Windows)

**ABBRs used:** CLI (Command‑Line Interface), venv (virtual environment), MD (Markdown)

This note shows several Windows‑native ways to launch a Python script **without** typing commands in PowerShell.

---

## Option A — Double‑click with a `.bat` wrapper (simple)

1) Create the folder for scripts (if not already):  
   `C:\Users\<you>\dev\cinderforge_e\scripts\`

2) Create this file (UTF‑8, CRLF):  
   **Path:** `scripts\run_my_tool.bat`  
   **Contents:**
   ```bat
   @echo off
   setlocal
   REM Activate venv if present
   if exist "%~dp0..\.venv-gpu\Scripts\python.exe" (
       set "PY=%~dp0..\.venv-gpu\Scripts\python.exe"
   ) else (
       set "PY=py"
   )
   "%PY%" "%~dp0my_tool.py" %*
   echo.
   pause
   ```

3) Place your Python file next to it: `scripts\my_tool.py`.  
   Double‑click `run_my_tool.bat` to run.

**Why it works:** The batch file calls the venv’s `python.exe` if found; otherwise it falls back to the system launcher (`py`).

---

## Option B — File association with `py.exe` (system launcher)

- Right‑click any `.py` → **Open with** → **Choose another app** → **Python** (launcher).  
- Now double‑clicking `.py` runs with whatever default Python is registered.

**Pitfall:** This may bypass your repo venv. Prefer the `.bat` wrapper for reproducible runs.

---

## Option C — `pythonw.exe` for GUI tools (no console window)

For scripts that open a window and don’t need a console:
- Copy your script to `scripts\my_gui_tool.pyw` (note `.pyw`).
- Create a `.bat` wrapper pointing to `pythonw.exe` in the venv:
  ```bat
  @echo off
  "%~dp0..\.venv-gpu\Scripts\pythonw.exe" "%~dp0my_gui_tool.pyw"
  ```

---

## Option D — Windows Task Scheduler (timed runs)

- **Action → Program:** `C:\Windows\py.exe`  
- **Add arguments:** `"C:\Users\<you>\dev\cinderforge_e\scripts\my_tool.py"`  
- **Start in:** `C:\Users\<you>\dev\cinderforge_e\scripts`

---

## Verify
- If your tool writes output, confirm the expected file appears.  
- If it imports local packages, ensure you installed the repo in editable mode: `pip install -e .`