import json
from pathlib import Path
from typing import List

def read_ndjson_tokens(path: str) -> List[str]:
    p = Path(path)
    toks: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "token_output" in obj:
                toks.append(str(obj["token_output"]))
            elif "text" in obj:
                toks.append(str(obj["text"]))
    return toks

def read_text_lines(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln for ln in (ln.strip() for ln in lines) if ln]
