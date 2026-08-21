"""Test local model verification policy."""

import idiolect.model


def test_mlx_runtime_fingerprint_covers_tokenization_and_platform(monkeypatch) -> None:
    """Check that inference identity records every material runtime layer."""
    versions = {
        "mlx-lm": "1",
        "mlx": "2",
        "transformers": "3",
        "tokenizers": "4",
        "jinja2": "5",
    }
    monkeypatch.setattr(idiolect.model, "version", versions.__getitem__)
    monkeypatch.setattr(idiolect.model.platform, "python_version", lambda: "3.14.0")
    monkeypatch.setattr(idiolect.model.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(idiolect.model.platform, "release", lambda: "25.0")
    monkeypatch.setattr(idiolect.model.platform, "machine", lambda: "arm64")

    value = idiolect.model.mlx_runtime_fingerprint()

    assert value == (
        "mlx-lm=1;mlx=2;transformers=3;tokenizers=4;jinja2=5;"
        "python=3.14.0;implementation=cpython;system=Darwin;"
        "release=25.0;machine=arm64"
    )
