# STEP TEMPLATE (use this shape)

[RUN COMMAND] — short purpose  
**ABBRs used:** CLI (Command‑Line Interface)  
```
Set-Location "C:\Users\<you>\dev\cinderforge_e"
```

[MAKE FILE] — short purpose  
**ABBRs used:** MD (Markdown)  
Path: `docs\howto\HOWTO - ... .md` (UTF‑8, CRLF)  
```
# Title
First use: PR (**Pull Request**) ...
```

[RUN COMMAND] — verify  
```
Test-Path "C:\Users\<you>\dev\cinderforge_e\docs\howto\HOWTO - ... .md"
```
Expected: `True`.