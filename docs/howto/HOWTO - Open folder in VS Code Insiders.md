# HOWTO — Open the **folder** in VS Code Insiders (not “open VS Code in the folder”)

**ABBRs used:** VS Code (Visual Studio Code), CLI (Command‑Line Interface)

Two correct ways:

### A) From File Explorer (mouse only)
1) Right‑click the repo folder → **Open with Code (Insiders)**.  
2) Confirm the left sidebar shows the folder name at the top.

### B) From PowerShell (**IN‑VSCODE** or **OUTSIDE**)
```
Set-Location "C:\Users\<you>\dev\cinderforge_e"
code-insiders .
```
**Why we care:** Copilot, the terminal, and file paths all key off the *workspace folder*. Opening a *single file* is not enough.