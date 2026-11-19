#!/usr/bin/env python3
"""Check the generated GNU CI compile database, not the CMake source text."""
import json
from pathlib import Path
import shlex
import sys

commands = json.loads(Path(sys.argv[1]).read_text())
checked = 0
for entry in commands:
    source = Path(entry["file"])
    if source.parent.name != "cmc" or source.suffix != ".c":
        continue
    args = entry.get("arguments") or shlex.split(entry["command"])
    standards = [arg for arg in args if arg.startswith("-std=")]
    if not standards or standards[-1] not in ("-std=gnu17", "-std=c17"):
        raise SystemExit(f"{source.name}: expected effective C17 mode, got {standards}; compiler flags may have been overwritten")
    checked += 1
if not checked:
    raise SystemExit("No CMC compilation commands found")
print(f"PASS: {checked} CMC compilation commands retain C17")
