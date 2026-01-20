import torch

from src.model import build_model


def test_build_model_forward_shape():
    model = build_model("temporal_avg_resnet18", pretrained=False)
    x = torch.randn(2, 4, 3, 32, 32)
    out = model(x)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_build_model_invalid_name():
    import pytest

    with pytest.raises(ValueError):
        build_model("unknown", pretrained=False)  # type: ignore[arg-type]
