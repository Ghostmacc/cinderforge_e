# HOWTO — Create & Save Files with Path Discipline (Windows)

**ABBRs used:** MD (Markdown)

1) Always start from the repo root: confirm with `git rev-parse --show-toplevel` (shows your repo path).  
2) Use **exact paths** in instructions; save, then verify via `Test-Path`.
3) When making a file I will give you: **Path, encoding (UTF‑8), EOL (CRLF), and full contents**.
4) After saving, run:  
```
Test-Path "C:\Users\<you>\dev\cinderforge_e\docs\howto\HOWTO - Example.md"
```
**Expected:** `True`.

**Tip:** Windows can hide case differences; keep filenames lowercase with underscores where possible.