#!/usr/bin/env python3
"""Extract Wiki-VISA images and normalize parquet rows without a GPU."""
import argparse
import glob
import hashlib
import io
import json
from pathlib import Path

if __package__:
    from .wiki_visa_data import DEFAULT_DATA_ROOT, DEFAULT_IMAGE_ROOT, expand_inputs, normalize_row
else:
    from wiki_visa_data import DEFAULT_DATA_ROOT, DEFAULT_IMAGE_ROOT, expand_inputs, normalize_row


from project_paths import project_path, relative_path


def index_image_ids(patterns):
    """Read IDs from filenames without loading gigabytes of vector JSON."""
    if not patterns:
        return None
    ids = set()
    for pattern in patterns:
        directories = sorted(glob.glob(str(project_path(pattern))))
        if not directories:
            raise FileNotFoundError(f"No index directories match: {pattern}")
        for directory in directories:
            nodes = list(Path(directory).glob("*.node"))
            if not nodes:
                raise ValueError(f"No .node files in {directory}")
            ids.update(path.stem for path in nodes)
    return ids


def in_partition(image_id, partition, validation_fraction):
    # Split by page ID so questions about the same page never cross partitions.
    value = int(hashlib.sha256(str(image_id).encode()).hexdigest()[:16], 16) / 2**64
    return partition == "all" or (value < validation_fraction) == (partition == "val")


def verl_row(sample, row, index, image_id):
    bbox = sample["bounding_box"]
    if isinstance(bbox, str):
        bbox = json.loads(bbox)
    question = row.get("question") or sample["query"].rsplit("Question: ", 1)[-1]
    return {
        "uid": sample["uid"], "data_source": "wiki-visa",
        "prompt": [{"role": "user", "content": sample["query"]}],
        "ability": "visual-retrieval",
        "reward_model": {"style": "rule", "ground_truth": sample["reference_answer"]},
        "extra_info": {"index": index, "question": str(question),
                       "reference_page": sample["reference_page"],
                       "ground_truth_citation": json.dumps(bbox), "bounding_box": json.dumps(bbox),
                       "page_id": str(image_id)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Raw HF parquet paths/globs")
    parser.add_argument("--output", default=str(DEFAULT_DATA_ROOT / "test.parquet"))
    parser.add_argument("--image_root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--index_dir", nargs="+", help="Keep only page IDs present in these index dirs/globs")
    parser.add_argument("--format", choices=["inference", "verl"], default="inference")
    parser.add_argument("--partition", choices=["all", "train", "val"], default="all")
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--skip_incomplete", action="store_true", help="Explicitly skip shards with unreadable parquet footers")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation_fraction must be between 0 and 1")
    selected_ids = index_image_ids(args.index_dir)
    paths = expand_inputs(args.input)
    output = project_path(args.output)
    if output in paths:
        parser.error("--output cannot overwrite an input file")
    if output.exists() and not args.overwrite:
        parser.error(f"Output exists: {output}; use --overwrite deliberately")

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    image_root = project_path(args.image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        parser.error(f"Partial output exists: {temporary}; inspect it before retrying")
    schema = pa.schema([
        ("uid", pa.string()), ("query", pa.string()),
        ("reference_answer", pa.string()), ("bounding_box", pa.string()),
        ("reference_page", pa.list_(pa.string())),
    ])
    if args.format == "verl":
        schema = pa.schema([
            ("uid", pa.string()), ("data_source", pa.string()),
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("ability", pa.string()),
            ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            ("extra_info", pa.struct([
                ("index", pa.int64()), ("question", pa.string()),
                ("reference_page", pa.list_(pa.string())),
                ("ground_truth_citation", pa.string()),
                ("bounding_box", pa.string()), ("page_id", pa.string()),
            ])),
        ])
    count = 0
    skipped = []
    exported_ids = set()
    try:
        with pq.ParquetWriter(temporary, schema) as writer:
            for path in paths:
                try:
                    parquet = pq.ParquetFile(path)
                except Exception as exc:
                    if args.skip_incomplete and isinstance(exc, pa.ArrowInvalid):
                        skipped.append(relative_path(path))
                        print(f"Skipping incomplete shard: {relative_path(path)} ({type(exc).__name__})", flush=True)
                        continue
                    raise ValueError(f"Cannot read {path}; its download may be incomplete") from exc
                index = 0
                for batch in parquet.iter_batches(batch_size=8):
                    rows = []
                    for row in batch.to_pylist():
                        sample = normalize_row(row, path, index)
                        index += 1
                        image_id = str(row.get("id") or (sample["reference_page"] or [sample["uid"]])[0])
                        if selected_ids is not None and image_id not in selected_ids:
                            continue
                        if not in_partition(image_id, args.partition, args.validation_fraction):
                            continue
                        image = row.get("image")
                        if isinstance(image, dict) and image.get("bytes"):
                            if image_id in (".", "..") or "/" in image_id or "\\" in image_id:
                                raise ValueError(f"Invalid image ID: {image_id!r}")
                            destination = image_root / f"{image_id}.jpg"
                            if not destination.exists():
                                with Image.open(io.BytesIO(image["bytes"])) as img:
                                    pending_image = destination.with_suffix(".jpg.partial")
                                    try:
                                        if img.format == "JPEG":
                                            pending_image.write_bytes(image["bytes"])
                                        else:
                                            img.convert("RGB").save(pending_image, format="JPEG", quality=95)
                                        pending_image.replace(destination)
                                    finally:
                                        pending_image.unlink(missing_ok=True)
                        exported_ids.add(image_id)
                        bbox = sample["bounding_box"]
                        if args.format == "verl":
                            rows.append(verl_row(sample, row, count, image_id))
                        else:
                            sample["bounding_box"] = bbox if isinstance(bbox, str) else json.dumps(bbox)
                            rows.append(sample)
                        count += 1
                        if args.max_samples and count >= args.max_samples:
                            break
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    if args.max_samples and count >= args.max_samples:
                        break
                print(f"Read {relative_path(path)}; selected {count} rows so far", flush=True)
                if args.max_samples and count >= args.max_samples:
                    break
        if count == 0:
            raise ValueError("No samples selected; check input downloads, index IDs and partition")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Prepared {count} samples: {output}")
    print(f"Images: {image_root}")
    print(f"Unique selected page IDs: {len(exported_ids)}; skipped shards: {len(skipped)}")


if __name__ == "__main__":
    main()
