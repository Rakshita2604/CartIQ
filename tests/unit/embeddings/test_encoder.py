from pathlib import Path

import numpy as np
import pytest

from cartiq.embeddings.encoder import HashingEncoder, format_input, l2_normalize, pool
from cartiq.embeddings.models import MODELS, ModelIntegrityError, Pooling, verify_model


def test_mean_pooling_ignores_padding() -> None:
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)

    assert pool(hidden, mask, Pooling.MEAN).tolist() == [[2.0, 3.0]]  # (1+3)/2, (2+4)/2


def test_cls_pooling_takes_the_first_token() -> None:
    hidden = np.array([[[5.0, 6.0], [1.0, 1.0]]], dtype=np.float32)

    assert pool(hidden, np.array([[1, 1]]), Pooling.CLS).tolist() == [[5.0, 6.0]]


def test_l2_normalize() -> None:
    vectors = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))

    assert vectors[0].tolist() == pytest.approx([0.6, 0.8])
    assert vectors[1].tolist() == [0.0, 0.0]  # zero vectors stay zero, no division error


@pytest.mark.parametrize(
    ("prefix", "text", "expected"),
    [("passage: ", "", "passage:"), ("query: ", "milk", "query: milk"), ("", "  milk ", "milk")],
)
def test_format_input_strips_surrounding_whitespace(prefix: str, text: str, expected: str) -> None:
    assert format_input(prefix, text) == expected


def test_model_specific_formatting_and_pooling() -> None:
    assert (MODELS["e5-small"].query_prefix, MODELS["e5-small"].document_prefix) == (
        "query: ",
        "passage: ",
    )
    assert MODELS["bge-small"].query_prefix.startswith("Represent this sentence")
    assert MODELS["bge-small"].document_prefix == ""
    assert (MODELS["minilm-l12"].query_prefix, MODELS["minilm-l12"].document_prefix) == ("", "")
    assert [MODELS[k].pooling for k in ("e5-small", "minilm-l12", "bge-small")] == [
        Pooling.MEAN,
        Pooling.MEAN,
        Pooling.CLS,
    ]
    assert [MODELS[k].max_length for k in ("e5-small", "minilm-l12", "bge-small")] == [
        512,
        128,
        512,
    ]


def test_identity_covers_revision_hashes_and_formatting() -> None:
    identity = MODELS["e5-small"].identity

    for part in (
        MODELS["e5-small"].revision,
        MODELS["e5-small"].onnx.sha256,
        "mean",
        "max512",
        "query: ",
    ):
        assert part in identity


def test_verify_model_refuses_missing_or_altered_files(tmp_path: Path) -> None:
    spec = MODELS["bge-small"]
    with pytest.raises(ModelIntegrityError, match="missing"):
        verify_model(spec, tmp_path)
    directory = spec.directory(tmp_path)
    (directory / "onnx").mkdir(parents=True)
    (directory / spec.onnx.path).write_bytes(b"not the model")
    (directory / spec.tokenizer.path).write_bytes(b"{}")
    with pytest.raises(ModelIntegrityError, match="sha256"):
        verify_model(spec, tmp_path)


def test_hashing_encoder_is_deterministic_and_normalised() -> None:
    encoder = HashingEncoder(dimension=32)

    a = encoder.encode_queries(["amul milk", "doodh"])
    b = encoder.encode_queries(["amul milk", "doodh"])

    assert np.array_equal(a, b)
    assert np.linalg.norm(a, axis=1) == pytest.approx([1.0, 1.0])
    assert encoder.identity == "hashing-v1|dim32"
