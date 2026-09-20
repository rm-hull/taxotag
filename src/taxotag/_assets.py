"""Resolve Gist model assets from a local directory or the Hugging Face Hub."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

REPOSITORY = "desert-ant-labs/gist"


def _get_revision() -> str:
    """Read revision from .model-revision file."""
    # Look for .model-revision in the package directory
    revision_file = Path(__file__).parent / ".model-revision"
    if revision_file.exists():
        return revision_file.read_text().strip()
    # Fallback to default
    return "v2.2.0"


REVISION = _get_revision()


@dataclass(frozen=True)
class AssetPaths:
    """Paths to the files required by the multilingual or English model."""

    tokenizer: Path
    embedding: Path
    embedding_meta: Path
    config: Path
    taxonomy: Path
    tflite: Path


def _filenames(variant: str) -> dict[str, str]:
    if variant not in {"multilingual", "english"}:
        raise ValueError("variant must be 'multilingual' or 'english'")
    prefix = "en/" if variant == "english" else ""
    return {
        "tokenizer": f"{prefix}gist_tokenizer.bin",
        "embedding": f"{prefix}gist_embedding.i8",
        "embedding_meta": f"{prefix}gist_embedding.json",
        "config": f"{prefix}gist_config.json",
        "taxonomy": f"{prefix}taxonomy.json",
        "tflite": f"{prefix}gist.tflite",
    }


def resolve_assets(
    directory: str | Path | None = None,
    variant: str = "multilingual",
) -> AssetPaths:
    """Resolve all required model files, using a local directory when complete."""

    filenames = _filenames(variant)
    if directory is not None:
        root = Path(directory)
        local_paths = {name: root / filename for name, filename in filenames.items()}
        if all(path.is_file() for path in local_paths.values()):
            return AssetPaths(**local_paths)

        root.mkdir(parents=True, exist_ok=True)
        resolved = {
            name: Path(
                hf_hub_download(
                    repo_id=REPOSITORY,
                    filename=filename,
                    revision=REVISION,
                    local_dir=root,
                )
            )
            for name, filename in filenames.items()
        }
        return AssetPaths(**resolved)

    resolved = {
        name: Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                filename=filename,
                revision=REVISION,
            )
        )
        for name, filename in filenames.items()
    }
    return AssetPaths(**resolved)
