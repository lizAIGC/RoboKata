"""Public launcher for the single-view Franka RoboKata recipe."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    'prepare': 'prepare.py', 'check-data': 'check_data.py',
    'demo-features': 'demo_features.py', 'semantic-features': 'semantic_features.py',
    'train': 'train_entry.py', 'infer': 'infer_checkpoint.py',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-command', action='store_true', help='Preview without importing GPU dependencies')
    parser.add_argument('--gpus', type=int, default=4, help='Training worker count; select devices with CUDA_VISIBLE_DEVICES')
    parser.add_argument('command', choices=ENTRYPOINTS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error('--gpus must be positive')
    forwarded = args.arguments
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    cmd = [sys.executable]
    if args.command == 'train':
        cmd += ['-m', 'torch.distributed.run', '--standalone', f'--nproc-per-node={args.gpus}']
    cmd.append(str(ROOT / 'roboicl' / ENTRYPOINTS[args.command]))
    if args.command == 'train' and not any(v in ('-h', '--help') for v in forwarded):
        recipe = json.loads((ROOT / 'configs/train.json').read_text(encoding='utf-8'))
        for key, value in recipe.items():
            if key != 'gpus':
                flag = '--' + key.replace('_', '-')
                if not any(arg == flag or arg.startswith(flag + '=') for arg in forwarded):
                    cmd += [flag, str(value)]
        if '--foresight-indexing' not in forwarded:
            cmd.append('--foresight-indexing')
    cmd += forwarded
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT / 'backend'), env.get('PYTHONPATH')]))
    if args.print_command:
        print('PYTHONPATH=' + shlex.quote(env['PYTHONPATH']) + ' ' + shlex.join(cmd))
        return 0
    return subprocess.call(cmd, env=env)

if __name__ == '__main__':
    raise SystemExit(main())
