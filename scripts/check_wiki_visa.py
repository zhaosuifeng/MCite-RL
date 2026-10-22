#!/usr/bin/env python3
"""Check local Wiki-VISA inputs and selected runtime prerequisites without downloading models."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_paths import project_path, relative_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['data', 'search', 'model', 'train'], default='data')
    parser.add_argument('--index_dir', default='corpus/wiki-visa/val_batch_0')
    parser.add_argument('--image_root', default='data/wiki-visa/images')
    parser.add_argument('--model', default='models/policy')
    parser.add_argument('--embed_model', default='models/colqwen2-v0.1-merged')
    parser.add_argument('--report_only', action='store_true', help='Report missing requirements without a nonzero exit')
    args = parser.parse_args()
    errors = []
    report = {'stage': args.stage, 'python': sys.version.split()[0]}
    import pyarrow.parquet as pq
    def parquet(name, required=False):
        p = project_path(name)
        try:
            q = pq.ParquetFile(p)
            report[name] = {'rows': q.metadata.num_rows, 'columns': q.schema_arrow.names}
            if q.metadata.num_rows == 0:
                raise ValueError('no rows')
            return q
        except Exception as exc:
            report[name] = {'error': type(exc).__name__}
            if required:
                errors.append(f'Missing or unreadable {name}')
            return None
    if args.stage in ('data', 'search'):
        q = parquet('data/wiki-visa/test.parquet', required=True) if args.stage == 'data' else None
        index = project_path(args.index_dir)
        ids = {p.stem for p in index.glob('*.node')}
        images = {p.stem for p in project_path(args.image_root).rglob('*.jpg')}
        missing = sorted(ids-images)
        report['index'] = relative_path(index)
        report['index_pages'] = len(ids)
        report['exported_images'] = len(images)
        report['index_images_missing'] = len(missing)
        if not ids:
            errors.append('No node index files found')
        if missing:
            errors.append(f'{len(missing)} indexed pages have no image; first IDs: {missing[:3]}')
        if q and 'reference_page' in q.schema_arrow.names:
            refs = {str(x) for batch in q.iter_batches(columns=['reference_page'])
                    for row in batch.to_pylist() for x in row['reference_page']}
            report['reference_pages_outside_index'] = len(refs-ids)
            if refs-ids:
                errors.append('Test rows include reference pages outside the selected retrieval index')
    if args.stage == 'train':
        partitions = {}
        for name in ('train', 'val'):
            q = parquet(f'data/wiki-visa/{name}.parquet', required=True)
            if q:
                needed = {'prompt','data_source','reward_model','extra_info'}
                if not needed.issubset(q.schema_arrow.names):
                    errors.append(f'{name}.parquet is not VERL format; run prepare-train')
                else:
                    rows = q.read(columns=['extra_info']).to_pylist()
                    partitions[name] = {row['extra_info'].get('page_id') for row in rows}
                if name == 'train' and q.metadata.num_rows < 2:
                    errors.append('The smoke configuration needs at least two train rows')
        if len(partitions) == 2:
            overlap = partitions['train'] & partitions['val']
            report['train_val_page_overlap'] = len(overlap)
            if overlap:
                errors.append('Training and validation share page IDs')
        if importlib.util.find_spec('flash_attn') is None:
            errors.append('flash-attn is missing; current FSDP worker explicitly uses flash_attention_2')
    if args.stage in ('search','model','train'):
        import torch
        report['cuda_available'] = torch.cuda.is_available()
        report['cuda_device_count'] = torch.cuda.device_count()
        if not report['cuda_available']:
            errors.append('CUDA is unavailable in this environment')
        model = project_path(args.embed_model if args.stage == 'search' else args.model)
        report['weights_dir'] = relative_path(model)
        if not (model/'config.json').is_file() or not (list(model.rglob('*.safetensors')) or list(model.rglob('pytorch_model*.bin'))):
            errors.append(f'Missing complete local model weights: {relative_path(model)}')
        if args.stage in ('model','train'):
            try:
                report['vllm_version'] = importlib.metadata.version('vllm')
            except importlib.metadata.PackageNotFoundError:
                errors.append('vLLM is missing; use the training Python environment')
    report['ready'] = not errors
    report['required_actions'] = errors
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors and not args.report_only:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
