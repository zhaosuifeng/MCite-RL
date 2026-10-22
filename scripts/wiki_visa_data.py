"""CPU-only Wiki-VISA loading shared by preparation and inference."""
from __future__ import annotations

import glob
import sys
import hashlib
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import project_path

DEFAULT_IMAGE_ROOT = Path("data/wiki-visa/images")
DEFAULT_DATA_ROOT = Path("data/wiki-visa")
PROMPT_FILE = PROJECT_ROOT / "prompts/wiki_visa_sft_gemini.txt"
USER_PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()
INVALID_ACTION_PROMPT = "Your previous action is invalid. " + USER_PROMPT.split("Question: ", 1)[0] + "Please try again."


def render_user_prompt(query: str, question: str | None = None) -> str:
    """Refresh old Wiki-VISA prompts without including labels in model input."""
    if question is None and "Question: " in query:
        question = query.rsplit("Question: ", 1)[1]
    return USER_PROMPT.format(question=question) if question else query



def expand_inputs(patterns: list[str]) -> list[Path]:
    files: dict[Path, None] = {}
    for pattern in patterns:
        matches = sorted(glob.glob(str(project_path(pattern))))
        if not matches:
            raise FileNotFoundError(f"No input files match: {pattern}")
        for name in matches:
            path = Path(name).resolve()
            if not path.is_file():
                raise ValueError(f"Expected a parquet file, not a directory: {path}")
            files[path] = None
    return list(files)


def normalize_row(row: dict, source: Path, index: int) -> dict:
    """Accept original HF rows and the project's prepared VERL-style rows.

    For original HF rows the embedded image is identified by ``id``; candidates
    are retained in the source dataset, not treated as reference-page labels.
    Ground-truth fields are saved for evaluation only, never added to the prompt.
    """
    extra = row.get("extra_info") or {}
    reward = row.get("reward_model") or {}
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        messages = [m.get("content", "") for m in prompt if isinstance(m, dict)]
        query = "\n".join(str(m) for m in messages)
    elif isinstance(prompt, str) and prompt.strip():
        query = prompt
    elif row.get("query") and "reference_answer" in row:
        query = str(row["query"])
    else:
        question = row.get("question") or row.get("query") or extra.get("question")
        if not question:
            raise ValueError(f"Missing question/prompt in {source.name}, row {index}")
        query = USER_PROMPT.format(question=question)
    query = render_user_prompt(query, row.get("question") or extra.get("question"))
    if not query.strip():
        raise ValueError(f"Empty prompt in {source.name}, row {index}")
    answer = reward.get("ground_truth")
    if answer is None:
        answer = row.get("short_answer", row.get("answer", row.get("reference_answer", "")))
    references = extra.get("reference_page", row.get("reference_page"))
    if references is None:
        image_id = row.get("id") or extra.get("file_name")
        references = [str(image_id)] if image_id is not None else []
    elif not isinstance(references, list):
        references = [references]
    bbox = extra.get("bounding_box", row.get("bounding_box"))
    identity = f"{source.name}:{index}:{query}"
    uid = row.get("uid") or hashlib.sha256(identity.encode()).hexdigest()[:24]
    return {
        "uid": str(uid), "query": query, "reference_answer": str(answer or ""),
        "bounding_box": bbox, "reference_page": [str(x) for x in references],
    }


def iter_samples(patterns: list[str], max_samples: int | None = None) -> Iterator[dict]:
    import pyarrow.parquet as pq

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    count = 0
    for path in expand_inputs(patterns):
        try:
            parquet = pq.ParquetFile(path)
        except Exception as exc:
            raise ValueError(f"Cannot read {path}; check that its download completed") from exc
        # Do not load embedded image bytes during inference.
        columns = [c for c in parquet.schema_arrow.names if c != "image"]
        index = 0
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                yield normalize_row(row, path, index)
                count += 1
                index += 1
                if max_samples is not None and count >= max_samples:
                    return
