"""Build-stage only: install a checksum-verified, fixed native Claude runtime."""
import hashlib
import os
from pathlib import Path
import platform
import re
import subprocess
import urllib.request

expected = os.environ.get('CLAUDE_SHA256', '')
if not re.fullmatch('[0-9a-f]{64}', expected):
    raise SystemExit('Supply official linux-x64 Claude 2.1.153 SHA256; never trust a computed download hash as its own reference')
if platform.system() != 'Linux' or platform.machine() not in {'x86_64', 'AMD64'}:
    raise SystemExit('This recipe is Linux x86_64 only')
target = Path('/opt/nice-sdk/bin/claude-pinned')
url = 'https://downloads.claude.ai/claude-code-releases/2.1.153/linux-x64/claude'
with urllib.request.urlopen(url, timeout=120) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit('Claude release checksum mismatch')
target.write_bytes(data)
target.chmod(0o755)
version = subprocess.check_output([str(target), '--version'], text=True, timeout=30).strip()
if version.split()[0] != '2.1.153':
    raise SystemExit('Claude executable version mismatch')
Path('/opt/nice-sdk/claude-sha256.txt').write_text(expected+'\n')
print(version)
