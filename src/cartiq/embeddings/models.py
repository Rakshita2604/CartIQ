"""Pinned embedding models: revisions, file hashes and model-specific input handling.

Every file is pinned by repository revision and SHA-256 and verified after download;
models live under ``<artifacts>/models/`` (gitignored). Settings below come from each
model's pinned repository files (``modules.json``, ``1_Pooling/config.json``,
``sentence_bert_config.json``) and model card:

- e5 (multilingual-e5-small): mean pooling; "query: " / "passage: " prefixes required.
- MiniLM (paraphrase-multilingual-MiniLM-L12-v2): mean pooling; no prefixes; max 128 tokens.
- bge (bge-small-en-v1.5): [CLS] pooling; queries get the card's retrieval instruction,
  passages none.

All encoders L2-normalise, so similarity is cosine (MiniLM's reference pipeline does not
normalise; cosine is unaffected). Inputs longer than ``max_length`` tokens (special tokens
included) are truncated from the end.
"""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Pooling(StrEnum):
    MEAN = "mean"  # attention-masked mean of the last hidden states
    CLS = "cls"  # the first ([CLS]) token's last hidden state


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    repo: str
    revision: str
    license: str
    onnx: ModelFile
    tokenizer: ModelFile
    pooling: Pooling
    max_length: int
    dimension: int
    query_prefix: str
    document_prefix: str
    reference_normalizes: bool  # whether the reference (sentence-transformers) pipeline does

    @property
    def identity(self) -> str:
        """Everything that determines this model's vectors for a given input text."""
        return (
            f"{self.repo}@{self.revision}|onnx:{self.onnx.sha256}|tok:{self.tokenizer.sha256}"
            f"|{self.pooling.value}|max{self.max_length}|l2"
            f"|q:{self.query_prefix!r}|d:{self.document_prefix!r}"
        )

    def directory(self, models_root: Path) -> Path:
        return models_root / self.key / self.revision[:12]


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="e5-small",
            repo="intfloat/multilingual-e5-small",
            revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            license="MIT",
            onnx=ModelFile(
                "onnx/model.onnx",
                "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
            ),
            tokenizer=ModelFile(
                "tokenizer.json",
                "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
            ),
            pooling=Pooling.MEAN,
            max_length=512,
            dimension=384,
            query_prefix="query: ",
            document_prefix="passage: ",
            reference_normalizes=True,
        ),
        ModelSpec(
            key="minilm-l12",
            repo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
            license="Apache-2.0",
            onnx=ModelFile(
                "onnx/model.onnx",
                "10f7a088420252b26caf819236ca2c9d2987afd0fc06fec7553b542a5655a05a",
            ),
            tokenizer=ModelFile(
                "tokenizer.json",
                "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
            ),
            pooling=Pooling.MEAN,
            max_length=128,
            dimension=384,
            query_prefix="",
            document_prefix="",
            reference_normalizes=False,
        ),
        ModelSpec(
            key="bge-small",
            repo="BAAI/bge-small-en-v1.5",
            revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            license="MIT",
            onnx=ModelFile(
                "onnx/model.onnx",
                "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
            ),
            # Not stored in LFS upstream (no published SHA-256): pinned from the download,
            # whose Git blob id was checked against the pinned revision.
            tokenizer=ModelFile(
                "tokenizer.json",
                "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            ),
            pooling=Pooling.CLS,
            max_length=512,
            dimension=384,
            query_prefix="Represent this sentence for searching relevant passages: ",
            document_prefix="",
            reference_normalizes=True,
        ),
    )
}


class ModelIntegrityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(spec: ModelSpec, models_root: Path) -> Path:
    """The model's directory, after checking every pinned file's SHA-256."""
    directory = spec.directory(models_root)
    for model_file in (spec.onnx, spec.tokenizer):
        path = directory / model_file.path
        if not path.is_file():
            raise ModelIntegrityError(f"{spec.key}: missing {path}; run `cartiq fetch-models`")
        actual = sha256_file(path)
        if actual != model_file.sha256:
            raise ModelIntegrityError(
                f"{spec.key}: {model_file.path} sha256 {actual} != pinned {model_file.sha256}"
            )
    return directory


def fetch_model(spec: ModelSpec, models_root: Path) -> Path:
    """Download the pinned files (if missing) and verify them."""
    from huggingface_hub import hf_hub_download  # network use is confined to fetching

    directory = spec.directory(models_root)
    for model_file in (spec.onnx, spec.tokenizer):
        hf_hub_download(
            repo_id=spec.repo,
            filename=model_file.path,
            revision=spec.revision,
            local_dir=directory,
        )
    return verify_model(spec, models_root)
