"""Text encoders: ONNX Runtime for real models, a hashing fake for tests.

``OnnxEncoder`` reproduces each model's reference pipeline: model-specific prefix, the
model's own tokenizer truncated to ``max_length`` tokens and padded per batch, the ONNX
graph's ``last_hidden_state``, the model's pooling, then L2 normalisation. Inference is
single-threaded with a fixed batch size and inputs in the given order, so the same texts
always give the same vectors on the same machine and library versions.
"""

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from cartiq.embeddings.models import ModelSpec, Pooling, verify_model

Vectors = npt.NDArray[np.float32]


class TextEncoder(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode_queries(self, texts: Sequence[str]) -> Vectors: ...

    def encode_documents(self, texts: Sequence[str]) -> Vectors: ...


class OnnxEncoder:
    def __init__(self, spec: ModelSpec, models_root: Path, *, batch_size: int = 32) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        directory = verify_model(spec, models_root)
        self._spec = spec
        self._batch_size = batch_size
        tokenizer = Tokenizer.from_file(str(directory / spec.tokenizer.path))
        pad_token = next(t for t in ("<pad>", "[PAD]") if tokenizer.token_to_id(t) is not None)
        tokenizer.enable_truncation(max_length=spec.max_length)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id(pad_token), pad_token=pad_token)
        self._tokenizer = tokenizer
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(directory / spec.onnx.path), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}

    @property
    def identity(self) -> str:
        return self._spec.identity

    @property
    def dimension(self) -> int:
        return self._spec.dimension

    def encode_queries(self, texts: Sequence[str]) -> Vectors:
        return self._encode([format_input(self._spec.query_prefix, text) for text in texts])

    def encode_documents(self, texts: Sequence[str]) -> Vectors:
        return self._encode([format_input(self._spec.document_prefix, text) for text in texts])

    def token_counts(self, texts: Sequence[str], *, query: bool) -> list[int]:
        """Tokens per text before truncation (special tokens included)."""
        prefix = self._spec.query_prefix if query else self._spec.document_prefix
        self._tokenizer.no_truncation()
        try:
            return [len(self._tokenizer.encode(format_input(prefix, t)).ids) for t in texts]
        finally:
            self._tokenizer.enable_truncation(max_length=self._spec.max_length)

    def _encode(self, texts: Sequence[str]) -> Vectors:
        out = np.zeros((len(texts), self._spec.dimension), dtype=np.float32)
        for start in range(0, len(texts), self._batch_size):
            batch = self._tokenizer.encode_batch(list(texts[start : start + self._batch_size]))
            ids = np.array([e.ids for e in batch], dtype=np.int64)
            mask = np.array([e.attention_mask for e in batch], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feed)[0].astype(np.float32)
            out[start : start + len(batch)] = pool(hidden, mask, self._spec.pooling)
        return l2_normalize(out)


def format_input(prefix: str, text: str) -> str:
    """Model prefix plus text, without surrounding whitespace. The reference tokenizer
    wrappers drop trailing whitespace while the raw ``tokenizers`` library would keep it as
    a token (e5 "passage: " + "" became an extra "▁"), so it is removed here."""
    return (prefix + text).strip()


def pool(
    hidden: npt.NDArray[np.float32], mask: npt.NDArray[np.int64], pooling: Pooling
) -> npt.NDArray[np.float32]:
    """[batch, tokens, dim] hidden states -> [batch, dim]."""
    if pooling is Pooling.CLS:
        return hidden[:, 0, :]
    weights = mask[:, :, None].astype(np.float32)
    summed = (hidden * weights).sum(axis=1)
    counts = np.clip(weights.sum(axis=1), 1e-9, None)
    return (summed / counts).astype(np.float32)


def l2_normalize(vectors: npt.NDArray[np.float32]) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)


_WORD = re.compile(r"\w+")


class HashingEncoder:
    """Deterministic stand-in for tests and CI (no network, no model files): a bag of
    hashed words and character trigrams, L2-normalised. Not meaningful for relevance."""

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def identity(self) -> str:
        return f"hashing-v1|dim{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_queries(self, texts: Sequence[str]) -> Vectors:
        return self._encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> Vectors:
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> Vectors:
        out = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                padded = f"#{word}#"
                for feature in [word, *(padded[i : i + 3] for i in range(len(padded) - 2))]:
                    bucket = int.from_bytes(hashlib.sha256(feature.encode()).digest()[:4], "big")
                    out[row, bucket % self._dimension] += 1.0
        return l2_normalize(out)
