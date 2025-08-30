# PITFALLS (read often)

- **Copilot overreach:** never accept multi‑file refactors blind; review staged diffs.  
- **Auth split (Desktop vs CLI):** clear Windows Credential Manager `github.com` entries if CLI prompts while Desktop works.  
- **Case sensitivity:** keep lowercase, snake_case; Windows hides conflicts seen on Linux.  
- **Save & path discipline:** after edits, `Ctrl+S` and `Test-Path` the exact file.  
- **Protected docs:** keep `docs/howto/*` and this file.  
- **Windows‑only scripts:** `.ps1` won’t run on Mac/Linux; prefer Python CLIs for portability.