"""Utilities for checking model installation status in local HuggingFace cache."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Common model weight file extensions
MODEL_WEIGHT_EXTENSIONS = {".bin", ".gguf", ".safetensors", ".pth", ".pt", ".ckpt"}


def _normalize_model_name(name: str) -> str:
    """Normalize model name for fuzzy matching.

    Strips "user." prefix, removes file extensions, normalizes separators to '-',
    strips known quantization suffixes, and collapses multiple separators.
    Produces a lowercase string where words are separated by single hyphens.
    """
    import re
    name = name.lower()
    # Remove "user." prefix
    name = re.sub(r'^user\.', '', name)
    # Remove file extension (only if actually at the end of the string)
    for ext in MODEL_WEIGHT_EXTENSIONS:
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    # Strip trailing '-gguf' (repo naming convention, not file extension)
    if name.endswith('-gguf'):
        name = name[:-5]
    # Normalize separators: replace .,_ with single hyphen
    name = re.sub(r'[_.]+', '-', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('-')
    # Remove quantization suffixes like -UD-IQ4_XS, -UD-Q4_K_XL, -IQ3_S, -Q4_0, etc.
    name = re.sub(r'-ud-[a-z0-9_-]+$', '', name)
    name = re.sub(r'-[A-Z]+[_-][A-Z0-9_]+$', '', name)
    return name


def is_llm_model_installed(model_identifier: str, cache_dir: Optional[Path] = None) -> bool:
    """
    Check if an LLM model (from the LLM API) is installed in the HF hub cache.

    Performs fuzzy matching between the model_identifier and cached model filenames:
    - Strips "user." prefix from model_identifier
    - Normalizes both strings (lowercase, remove separators like -_.)
    - Handles sharded models and quantization suffixes

    Args:
        model_identifier: Model ID from LLM API (e.g., "Qwen3.6-35B-A3B", "user.Llama-4-Scout-17B-16E-Instruct.gguf")
        cache_dir: Optional explicit hub cache dir

    Returns:
        True if a matching model file is found in any cached repo
    """
    target_norm = _normalize_model_name(model_identifier)

    if cache_dir is None:
        dirs_to_scan = _find_hf_cache_dirs()
    else:
        dirs_to_scan = [cache_dir]

    for snapshots_dir in _scan_snapshots_dirs(dirs_to_scan):
        try:
            for snapshot_dir in snapshots_dir.iterdir():
                if not snapshot_dir.is_dir():
                    continue
                for file in snapshot_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in MODEL_WEIGHT_EXTENSIONS:
                        file_norm = _normalize_model_name(file.name)
                        # Check if target is a substring of filename or vice versa
                        if target_norm in file_norm or file_norm.startswith(target_norm) or target_norm.startswith(file_norm):
                            return True
        except (PermissionError, OSError):
            continue

    return False


def list_installed_llm_models(cache_dir: Optional[Path] = None) -> List[Dict[str, object]]:
    """Return all installed LLM models detected in the HF hub cache.

    Filters out known non-LLM repos (e.g., whisper.cpp, sentence-transformers,
    nomic-embed, etc.) and groups files by their normalized model identifier.

    Returns:
        List of dicts: [{"model_id": str, "repo_id": str, "filenames": List[str], "size_gb": float}]
    """
    if cache_dir is None:
        dirs_to_scan = _find_hf_cache_dirs()
    else:
        dirs_to_scan = [cache_dir]

    models_by_id: dict[str, dict] = {}

    # Non-LLM repo prefixes/keywords to skip
    NON_LLM_PATTERNS = [
        "whisper", "faster-whisper", "ggerganov",
        "sentence-transform", "nomic-embed", "llmlingua",
        "bge-reranker", "docling", "onnx", "mmproj",  # mmproj files are vision projectors
    ]

    for snapshots_dir in _scan_snapshots_dirs(dirs_to_scan):
        try:
            repo_dir = snapshots_dir.parent
            repo_name = repo_dir.name
            repo_id = _parse_repo_id(repo_name)
            if not repo_id:
                continue

            # Skip known non-LLM repos
            if any(patt in repo_id.lower() for patt in NON_LLM_PATTERNS):
                continue

            for snapshot_dir in snapshots_dir.iterdir():
                if not snapshot_dir.is_dir():
                    continue
                for file in snapshot_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in MODEL_WEIGHT_EXTENSIONS:
                        try:
                            size = file.stat().st_size
                        except OSError:
                            size = 0

                        norm = _normalize_model_name(file.stem)
                        repo_key = _normalize_model_name(repo_id)
                        model_id = f"{repo_key}/{norm}" if '/' not in norm else norm

                        entry = models_by_id.setdefault(model_id, {
                            "model_id": model_id,
                            "repo_id": repo_id,
                            "filenames": [],
                            "size_gb": 0.0,
                        })
                        entry["filenames"].append(file.name)
                        entry["size_gb"] += size / 1e9
        except (PermissionError, OSError):
            continue

    return list(models_by_id.values())


def get_hf_cache_dir() -> Path:
    """
    Get the HuggingFace hub cache directory (contains models--*).

    Priority:
      1. HF_HUB_CACHE          — direct absolute path to cache dir
      2. HUGGINGFACE_HUB_CACHE — direct absolute path to cache dir
      3. HF_HOME               — base dir, cache at $HF_HOME/hub
      4. HUGGINGFACE_HOME      — base dir, cache at $HUGGINGFACE_HOME/hub
      5. Default               — ~/.cache/huggingface/hub

    Returns:
        Path to the HF hub cache directory (always returns a path,
        even if it does not yet exist on disk)
    """
    # 1. HF_HUB_CACHE (newer name)
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)

    # 2. HUGGINGFACE_HUB_CACHE
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)

    # 3 & 4. Base directories — hub is a subdirectory named "hub"
    base = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HOME")
    if base:
        return Path(base) / "hub"

    # 5. Default
    return Path.home() / ".cache" / "huggingface" / "hub"


def _find_hf_cache_dirs() -> list[Path]:
    """Return all known HuggingFace cache locations to scan.

    Includes the primary hub dir plus common fallback locations that may
    contain model weight files (e.g. when HF_HUB_CACHE points to a custom
    path or models were downloaded via older huggingface_hub versions).
    """
    candidates: list[Path] = []
    try:
        candidates.append(get_hf_cache_dir())
    except Exception:
        pass

    # Fallback locations that may contain models--*/snapshots/
    for base in (
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".local" / "share" / "huggingface",
    ):
        candidates.append(base / "hub")

    return candidates


def _scan_snapshots_dirs(dirs: list[Path]) -> list[Path]:
    """Yield snapshot directories from a list of hub cache locations."""
    for hub_dir in dirs:
        if not hub_dir.exists() or not hub_dir.is_dir():
            continue
        try:
            for item in hub_dir.iterdir():
                if item.name.startswith("models--"):
                    snapshots = item / "snapshots"
                    if snapshots.exists() and snapshots.is_dir():
                        yield snapshots
        except PermissionError:
            logger.warning("Permission denied scanning %s", hub_dir)
        except OSError as exc:
            logger.debug("Error scanning %s: %s", hub_dir, exc)


def _parse_repo_id(name: str) -> str | None:
    """Derive repo_id from a 'models--*' directory name."""
    if not name.startswith("models--"):
        return None
    repo_part = name[len("models--") :]
    # Handle "owner--repo" → "owner/repo"
    if "--" in repo_part:
        return repo_part.replace("--", "/", 1)
    return repo_part if repo_part else None


def find_model_in_cache(
    repo_id: str,
    filename: str,
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Search the HuggingFace hub cache for a specific model file.

    Args:
        repo_id: HuggingFace repo ID (e.g. "ggerganov/whisper.cpp")
        filename: Filename to look for (e.g. "ggml-large-v3-turbo.bin")
        cache_dir: Optional explicit hub cache dir (default: from get_hf_cache_dir())

    Returns:
        Path to the model file if found in any snapshot, None otherwise
    """
    if cache_dir is None:
        cache_dir = get_hf_cache_dir()

    # Build the models directory name: "models--{repo_id.replace('/', '--')}"
    models_subdir = f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = cache_dir / models_subdir / "snapshots"

    if not snapshots_dir.exists():
        return None

    for snapshot_dir in snapshots_dir.iterdir():
        if not snapshot_dir.is_dir():
            continue
        candidate = snapshot_dir / filename
        if candidate.exists():
            return candidate

    return None


def is_model_installed(
    repo_id: str,
    filename: str,
    cache_dir: Optional[Path] = None,
) -> bool:
    """True if the model file exists in the HF hub cache."""
    return find_model_in_cache(repo_id, filename, cache_dir) is not None


def is_whisper_model_installed(model_name: str, cache_dir: Optional[Path] = None) -> bool:
    """Check if a Whisper.cpp model is installed in the HF hub cache.

    Handles model names from Lemonade (e.g., "Whisper-Large-v3-Turbo", "Whisper-Small")
    and matches them against HF cache filenames (e.g., ggml-large-v3-turbo.bin).

    Args:
        model_name: Whisper model name (may include "Whisper-" prefix)
        cache_dir: Optional explicit hub cache dir

    Returns:
        True if the model file is found in the ggml/gguf repo
    """
    import re

    # Normalize model name: lowercase, strip "whisper-" prefix
    mname = model_name.lower()
    if mname.startswith("whisper-"):
        mname = mname[len("whisper-"):]
    # Remove any separators
    mname_norm = re.sub(r'[-_.\s]+', '', mname)
    # If ends with "turbo", also try without it
    has_turbo = mname_norm.endswith("turbo")
    base_norm = mname_norm[:-len("turbo")] if has_turbo else mname_norm

    if cache_dir is None:
        dirs_to_scan = _find_hf_cache_dirs()
    else:
        dirs_to_scan = [cache_dir]

    for snapshots_dir in _scan_snapshots_dirs(dirs_to_scan):
        try:
            repo_dir = snapshots_dir.parent
            repo_name = repo_dir.name
            repo_id = _parse_repo_id(repo_name)
            if repo_id != "ggerganov/whisper.cpp":
                continue

            for snapshot_dir in snapshots_dir.iterdir():
                if not snapshot_dir.is_dir():
                    continue
                for file in snapshot_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in MODEL_WEIGHT_EXTENSIONS:
                        fname = file.stem.lower()
                        if fname.startswith("ggml-"):
                            fname = fname[len("ggml-"):]
                        fname_norm = re.sub(r'[-_.\s]+', '', fname)

                        if mname_norm == fname_norm:
                            return True
                        if has_turbo and base_norm == fname_norm:
                            return True
        except (PermissionError, OSError):
            continue

    return False


def is_llm_model_installed(model_identifier: str, cache_dir: Optional[Path] = None) -> bool:
    """
    Check if an LLM model (from LLM API) is installed in the HF hub cache.

    Performs robust fuzzy matching between model_identifier and cached model files.
    Handles "user." prefix, file extensions, quantization suffixes, and sharding.
    Also matches by repo if identifier contains a '/'.

    Args:
        model_identifier: Model ID from LLM API (e.g., "Qwen3.6-35B-A3B",
                          "unsloth/Qwen3.6-35B-A3B", "user.Llama-4-Scout.gguf")
        cache_dir: Optional explicit hub cache dir

    Returns:
        True if a matching model file is found
    """
    import re

    # Split into optional repo prefix and model name
    # Identifier can be "repo/model" or just "model"
    if '/' in model_identifier:
        repo_part, model_part = model_identifier.split('/', 1)
    else:
        repo_part = None
        model_part = model_identifier

    target_norm = _normalize_model_name(model_part)

    if cache_dir is None:
        dirs_to_scan = _find_hf_cache_dirs()
    else:
        dirs_to_scan = [cache_dir]

    for snapshots_dir in _scan_snapshots_dirs(dirs_to_scan):
        try:
            # Get the repo_id for this snapshots dir
            repo_dir = snapshots_dir.parent
            repo_name = repo_dir.name
            repo_id = _parse_repo_id(repo_name)
            if not repo_id:
                continue

            # If identifier specified a repo, only match that repo
            if repo_part is not None:
                repo_norm = _normalize_model_name(repo_part)
                if repo_norm not in _normalize_model_name(repo_id):
                    continue  # repo mismatch

            for snapshot_dir in snapshots_dir.iterdir():
                if not snapshot_dir.is_dir():
                    continue
                for file in snapshot_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in MODEL_WEIGHT_EXTENSIONS:
                        file_norm = _normalize_model_name(file.name)
                        # Strong match: filename contains the target as a complete token
                        pattern = r'(?:^|[-_.])' + re.escape(target_norm) + r'(?:$|[-_.])'
                        if re.search(pattern, file_norm):
                            return True
        except (PermissionError, OSError):
            continue

    return False


def list_installed_llm_models(cache_dir: Optional[Path] = None) -> List[Dict[str, object]]:
    """Return all installed LLM models detected in the HF hub cache.

    Returns:
        List of dicts: [{"model_id": str, "repo_id": str, "filenames": List[str], "size_gb": float}]
    """
    if cache_dir is None:
        dirs_to_scan = _find_hf_cache_dirs()
    else:
        dirs_to_scan = [cache_dir]

    models_by_id: dict[str, dict] = {}

    for snapshots_dir in _scan_snapshots_dirs(dirs_to_scan):
        try:
            # Derive repo_id from the parent of snapshots_dir (models--* directory)
            repo_dir = snapshots_dir.parent
            repo_name = repo_dir.name
            repo_id_full = _parse_repo_id(repo_name)
            if not repo_id_full:
                continue

            # Skip known non-LLM repos
            NON_LLM_PATTERNS = [
                "whisper", "faster-whisper", "ggerganov",
                "sentence-transform", "nomic-embed", "llmlingua",
                "bge-reranker", "docling", "onnx", "mmproj",
            ]
            if any(patt in repo_id_full.lower() for patt in NON_LLM_PATTERNS):
                continue

            for snapshot_dir in snapshots_dir.iterdir():
                if not snapshot_dir.is_dir():
                    continue
                for file in snapshot_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in MODEL_WEIGHT_EXTENSIONS:
                        try:
                            size = file.stat().st_size
                        except OSError:
                            size = 0

                        norm = _normalize_model_name(file.stem)
                        repo_key = _normalize_model_name(repo_id_full)
                        model_id = f"{repo_key}/{norm}" if '/' not in norm else norm

                        entry = models_by_id.setdefault(model_id, {
                            "model_id": model_id,
                            "repo_id": repo_id_full,
                            "filenames": [],
                            "size_gb": 0.0,
                        })
                        entry["filenames"].append(file.name)
                        entry["size_gb"] += size / 1e9
        except (PermissionError, OSError):
            continue

    return list(models_by_id.values())


def _normalize_model_name(name: str) -> str:
    """Normalize model name for fuzzy matching.

    Strips "user." prefix, removes file extensions, normalizes separators to '-',
    strips known quantization suffixes, and collapses multiple separators.
    Produces a lowercase string where words are separated by single hyphens.
    """
    import re
    name = name.lower()
    # Remove "user." prefix
    name = re.sub(r'^user\.', '', name)
    # Remove file extension
    name = re.sub(r'\.(gguf|bin|safetensors|pth|pt|ckpt)(\b|$)', '', name)
    # Strip trailing '-gguf' (common repo convention, not a file extension)
    if name.endswith('-gguf'):
        name = name[:-5]
    # Normalize separators: replace .,_,- with single hyphen
    name = re.sub(r'[_.]+', '-', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('-')
    # Remove quantization suffixes like -UD-IQ4_XS, -UD-Q4_K_XL, -IQ3_S, -Q4_0, etc.
    name = re.sub(r'-ud-[a-z0-9_-]+$', '', name)
    name = re.sub(r'-[A-Z]+[_-][A-Z0-9_]+$', '', name)
    return name


def list_installed_models(
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """
    Scan the HuggingFace hub cache (and known fallback locations) for all
    installed model weight files.

    This is a robust, app-level method that does NOT depend on any SDK
    being installed — it directly scans the HF cache directory tree to
    discover every model file with a recognised weight extension
    (.bin, .gguf, .safetensors, .pth, .pt, .ckpt).

    Args:
        cache_dir: Optional explicit hub cache dir. If None,
                   ``get_hf_cache_dir()`` is used as the primary location,
                   and fallback locations are also scanned.

    Returns:
        List of dicts with keys:
            repo_id     — HuggingFace repo ID (e.g. "ggerganov/whisper.cpp")
            filename    — file name (e.g. "ggml-large-v3.bin")
            path        — absolute Path to the file
            size_bytes  — file size in bytes
    """
    # Build list of hub directories to scan
    if cache_dir is not None:
        dirs_to_scan: list[Path] = [cache_dir]
    else:
        dirs_to_scan = _find_hf_cache_dirs()

    results: list[dict] = []
    seen: set[Path] = set()  # avoid duplicates when multiple dirs overlap

    for snapshots_dir in _scan_snapshots_dirs(dirs_to_scan):
        try:
            for snapshot_dir in snapshots_dir.iterdir():
                if not snapshot_dir.is_dir():
                    continue
                for file in snapshot_dir.iterdir():
                    if (
                        file.is_file()
                        and file.suffix.lower() in MODEL_WEIGHT_EXTENSIONS
                        and file not in seen
                    ):
                        try:
                            size = file.stat().st_size
                        except OSError:
                            size = 0
                        # Derive repo_id from parent chain
                        # file -> snapshot_dir (commit hash) -> snapshots_dir (models--*)
                        repo_dir = snapshots_dir.parent
                        repo_name = repo_dir.name
                        repo_id = _parse_repo_id(repo_name)
                        if repo_id is None:
                            continue
                        seen.add(file)
                        results.append({
                            "repo_id": repo_id,
                            "filename": file.name,
                            "path": file,
                            "size_bytes": size,
                        })
        except PermissionError:
            logger.warning("Permission denied scanning snapshots %s", snapshots_dir)
        except OSError as exc:
            logger.debug("Error scanning snapshots %s: %s", snapshots_dir, exc)

    return results
