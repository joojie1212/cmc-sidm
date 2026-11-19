#!/usr/bin/env python3
"""Run a build and surface compiler errors in the GitHub job annotations."""
import subprocess
import sys

errors = []
with subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
    for line in process.stdout:
        print(line, end='', flush=True)
        if 'error:' in line.lower() or 'undefined reference' in line.lower():
            errors.append(line.strip())
    result = process.wait()
if result:
    for message in errors[-10:] or [f'Build exited with code {result}; see the build step log.']:
        message = message.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
        print(f'::error title=CMC build::{message}')
sys.exit(result)
