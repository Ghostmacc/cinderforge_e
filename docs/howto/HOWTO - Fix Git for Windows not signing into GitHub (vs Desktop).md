# HOWTO — Fix Git for Windows not signing into GitHub (while Desktop works)

**ABBRs used:** CLI (Command‑Line Interface), PAT (Personal Access Token)

1) Show current identity/helper (PowerShell):
```
git config --global user.name
git config --global user.email
git config --global credential.helper
```
Expect: your name/email; helper should be **manager-core**.

2) Enforce sane defaults:
```
git config --global user.name "<Your Name>"
git config --global user.email "you@yourmail.com"
git config --global credential.helper manager-core
git config --global core.longpaths true
```

3) Clear stale creds (Windows):
- **Start → Credential Manager → Windows Credentials →** remove **github.com** entries.

4) Ensure HTTPS (so the helper can prompt):
```
git remote -v
git remote set-url origin https://github.com/<owner>/<repo>.git   # if needed
```

5) Trigger fresh auth:
```
git fetch origin
```
A browser/GUI prompt appears → sign into the same account used in Desktop.

6) Verify:
```
git ls-remote origin
```
Shows refs; no second prompt.

**Tip (multiple accounts):**
```
git config --global credential.useHttpPath true
```
Stores per‑repo credentials rather than one global entry.