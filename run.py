"""Run Tactile-ETA on a UniVTAC manipulation task."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
TASKS = ('insert_tube', 'lift_can', 'lift_bottle', 'pull_out_key')


def config_for(task: str, seed: int) -> dict:
    config = yaml.safe_load((ROOT / 'configs/tasks' / f'{task}.yaml').read_text())
    config['seed'] = seed
    config['seeds'] = [seed]
    config['demonstration_package'] = str((ROOT / config['demonstration_package']).resolve())
    return config


def check_demonstrations(task: str) -> dict:
    config = config_for(task, 1000000)
    package = Path(config['demonstration_package'])
    projection = json.loads((package / 'expert_demonstrations.json').read_text())
    for image in projection['images']:
        with Image.open(package / image['path']) as frame:
            frame.verify()
    return {'task': task, 'demonstrations': len(projection['text']['examples']),
            'images': len(projection['images'])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, default='insert_tube')
    parser.add_argument('--seed', type=int, default=1000000)
    parser.add_argument('--check', action='store_true', help='Check expert images without running an episode')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    if args.check:
        print(json.dumps([check_demonstrations(task) for task in TASKS], indent=2))
        return 0

    runtime = Path(os.environ['UNIVTAC_RUNTIME_PYTHON']).expanduser().resolve()
    source = Path(os.environ['UNIVTAC_SOURCE_ROOT']).expanduser().resolve()
    if not (source / 'envs/utils/env_parser.py').is_file():
        parser.error('UNIVTAC_SOURCE_ROOT must point to the UniVTAC Isaac51 checkout')
    if not runtime.is_file():
        parser.error('UNIVTAC_RUNTIME_PYTHON must point to the Isaac Sim Python executable')
    codex = os.environ.get('UNIVTAC_CODEX_BIN') or shutil.which('codex')
    if not codex:
        parser.error('Install Codex CLI or set UNIVTAC_CODEX_BIN')
    os.environ['UNIVTAC_CODEX_BIN'] = codex
    os.environ['UNIVTAC_CONTROLLER_PYTHON'] = sys.executable
    auth = Path(os.environ.get('UNIVTAC_CODEX_AUTH_FILE', str(Path.home() / '.codex/auth.json'))).expanduser()
    if not auth.is_file():
        parser.error('Sign in with Codex CLI or set UNIVTAC_CODEX_AUTH_FILE')
    os.environ['UNIVTAC_CODEX_AUTH_FILE'] = str(auth.resolve())

    config = config_for(args.task, args.seed)
    parent = args.output_root.resolve() / args.task / str(args.seed)
    episode_dir = parent / 'episode'
    if episode_dir.exists():
        parser.error(f'Episode already exists: {episode_dir}. Choose another --output-root.')
    parent.mkdir(parents=True, exist_ok=True)
    config_file = parent / 'config.yaml'
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    lane = SimpleNamespace(mode='batch', config=config_file, runtime_python=runtime,
                           source_root=source, coordinator=None)
    os.chdir(ROOT)
    from scripts.univtac.run_autonomous_insert_hole import run_episode

    result = run_episode(lane, config, args.seed, episode_dir)
    print(json.dumps({'episode_dir': str(episode_dir), 'status': result.get('status'),
                      'task_success': result.get('task_success'),
                      'termination': result.get('termination'),
                      'infrastructure_error': result.get('infrastructure_error')}, indent=2))
    return 0 if result.get('evaluable') else 2


if __name__ == '__main__':
    raise SystemExit(main())
