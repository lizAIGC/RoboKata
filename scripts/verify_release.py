"""Dependency-free packaging checks; does not run the model."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'configs/source-manifest.json').read_text())
for name, digest in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
files = list(root.rglob('*.py'))
for path in files:
    ast.parse(path.read_text(), filename=str(path))
launcher = root / 'scripts/robokata.py'
def preview(*args):
    result = subprocess.run([sys.executable, str(launcher), '--print-command', *args], check=True, capture_output=True, text=True, cwd='/tmp')
    return shlex.split(result.stdout.strip())
cmd = preview('--gpus', '2', 'train', '--learning-rate', '0.0001', '--work', '/tmp/path with spaces', '--epochs=1')
assert '--nproc-per-node=2' in cmd
assert cmd.count('--learning-rate') == 1
assert cmd[cmd.index('--learning-rate') + 1] == '0.0001'
assert '--epochs=1' in cmd and '--epochs' not in cmd
assert cmd[cmd.index('--work') + 1] == '/tmp/path with spaces'
assert '--foresight-indexing' in cmd
cmd = preview('infer', '--checkpoint', '/tmp/example.safetensors', '--pair-index', '3')
assert not any('torch.distributed' in item for item in cmd)
assert cmd[cmd.index('--pair-index') + 1] == '3'
assert not (root / 'baselines').exists() and not (root / 'baseline').exists()
print(json.dumps({'source_files_verified':len(manifest['files']), 'python_files_parsed':len(files), 'launcher_checks':'passed', 'gpu_training':'not run', 'gpu_inference':'not run'}, indent=2))
