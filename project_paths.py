"""Portable project paths; relative paths always start at this file's directory.

No training or inference dependencies are imported here. Absolute user overrides
and remote URIs remain supported; model repository IDs are not filesystem paths.
"""
from pathlib import Path
import os
import re

PROJECT_ROOT = Path(__file__).resolve().parent


def is_uri(value):
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", str(value)))


def project_path(value):
    path = Path(value).expanduser()
    # Preserve resource symlinks so stored paths remain relative to this checkout.
    return Path(os.path.abspath(path if path.is_absolute() else PROJECT_ROOT / path))


def file_reference(value):
    return str(value) if is_uri(value) else str(project_path(value))


def relative_path(value):
    """Store paths inside the project portably, preserving external overrides."""
    if is_uri(value):
        return str(value)
    path = project_path(value)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def model_reference(value):
    value = str(value)
    if is_uri(value):
        return value
    # Bare org/model identifiers stay available for Hugging Face downloads.
    if (value.startswith(("/", "~", "./", "../", "models/", "checkpoints/"))
            or (PROJECT_ROOT / value).exists()):
        return str(project_path(value))
    return value


_RECORD_PATH_KEYS = {
    "path", "source_path", "source_crop_path", "image_file", "image_path",
    "file_path", "retrieved_image_path", "last_search_path", "last_search_image_path",
    "input_file", "inference_file", "source_image", "result_image", "debug_last_search_path",
}
_RECORD_PATH_LISTS = {"reference_image_paths", "retrievaled_images", "retrieved_images"}


def record_paths(record, *, relative):
    """Convert only filesystem fields, never prompts, IDs, URLs or base64 images."""
    convert = relative_path if relative else file_reference
    if isinstance(record, list):
        return [record_paths(item, relative=relative) for item in record]
    if not isinstance(record, dict):
        return record
    result = {}
    for key, value in record.items():
        if key in _RECORD_PATH_KEYS and isinstance(value, str) and value:
            result[key] = convert(value)
        elif key in _RECORD_PATH_LISTS and isinstance(value, list):
            result[key] = [convert(item) if isinstance(item, str) and item
                           else record_paths(item, relative=relative) for item in value]
        else:
            result[key] = record_paths(value, relative=relative)
    return result


def normalize_config_paths(config):
    """Resolve filesystem options before Hydra/Ray workers change directories."""
    from omegaconf import ListConfig, OmegaConf

    file_keys = (
        "data.train_files", "data.val_files", "data.path", "data.output_path",
        "data.cache_dir", "trainer.default_local_dir", "trainer.resume_from_path",
        "trainer.resume_path", "trainer.validation_data_dir", "trainer.rollout_data_dir",
        "custom_reward_function.path", "algorithm.rollout_data_dir",
    )
    model_keys = (
        "actor_rollout_ref.model.path", "critic.model.path", "critic.model.tokenizer_path",
        "reward_model.model.path", "reward_model.model.input_tokenizer",
        "model.path", "model.partial_pretrain", "model.tokenizer_path",
    )
    # Snapshot interpolations before updates so aliases are resolved consistently.
    updates = []
    for keys, convert in ((file_keys, file_reference), (model_keys, model_reference)):
        for key in keys:
            value = OmegaConf.select(config, key)
            if value is None or isinstance(value, bool) or value == "":
                continue
            if isinstance(value, (list, tuple, ListConfig)):
                value = [convert(item) for item in value]
            else:
                value = convert(value)
            updates.append((key, value))
    for key, value in updates:
        OmegaConf.update(config, key, value, merge=False)
    return config
