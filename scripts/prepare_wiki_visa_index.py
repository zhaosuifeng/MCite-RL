#!/usr/bin/env python3
"""Create a portable retrieval index view containing only pages with available images."""
import argparse
import glob
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_paths import project_path, relative_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs='+', required=True, help='Existing .node directories/globs')
    parser.add_argument('--image_root', default='data/wiki-visa/images')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = project_path(args.output)
    images = {p.stem for p in project_path(args.image_root).rglob('*.jpg')}
    sources = {}
    for pattern in args.source:
        for name in sorted(glob.glob(str(project_path(pattern)))):
            directory = project_path(name)
            if directory == output:
                raise ValueError('Output cannot be a source index')
            for node in sorted(directory.glob('*.node')):
                if node.stem in images:
                    if node.name in sources and sources[node.name] != node:
                        raise ValueError(f'Duplicate page index: {node.name}')
                    sources[node.name] = node
    if not sources:
        raise ValueError('No source index pages have extracted images')
    # Preflight before creating links, including stale files from an earlier view.
    if output.exists():
        for target in output.glob('*.node'):
            source = sources.get(target.name)
            if source is None or not target.is_symlink() or target.resolve() != source.resolve():
                raise ValueError(f'Existing index view differs: {target}; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        target = output/name
        if not target.is_symlink():
            target.symlink_to(os.path.relpath(source, output))
    manifest = {'pages': len(sources), 'sources': {name: relative_path(p) for name,p in sources.items()},
                'embedding_model': 'unknown: use the original encoder or rebuild vectors with your chosen encoder'}
    (output/'index-view.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(sources)} pages available in {relative_path(output)}; existing vectors preserved')


if __name__ == '__main__':
    main()
