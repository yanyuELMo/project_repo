import pytest
import torch
import torch.nn as nn

from src.train import _maybe_init_wandb, _sample_indices, _summarize_profile, evaluate


class _DummyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch = x.shape[0]
        logits = torch.linspace(5.0, -5.0, steps=batch, device=x.device)
        return logits


def test_evaluate_perfect_clip_metrics():
    model = _DummyModel()
    batch = {
        "x": torch.zeros((2, 1, 3, 4, 4)),
        "y": torch.tensor([1.0, 0.0]),
        "video_id": ["v1", "v2"],
    }
    loader = [batch]
    metrics = evaluate(
        model,
        loader,
        device=torch.device("cpu"),
        agg="max",
        clip_threshold=0.5,
        video_threshold=0.5,
        ratio_threshold=0.5,
        use_video_metrics=False,
    )

    assert metrics["clip_acc"] == 1.0
    assert metrics["clip_f1"] == 1.0
    assert metrics["clip_auc"] == 1.0
    assert metrics["video_auc"] is None


def test_evaluate_video_metrics_ratio_agg():
    model = _DummyModel()
    loader = [
        {
            "x": torch.zeros((3, 1, 3, 4, 4)),
            "y": torch.tensor([1.0, 0.0, 0.0]),
            "video_id": ["v1", "v1", "v2"],
        }
    ]
    metrics = evaluate(
        model,
        loader,
        device=torch.device("cpu"),
        agg="ratio",
        clip_threshold=0.5,
        video_threshold=0.5,
        ratio_threshold=0.5,
        use_video_metrics=True,
    )

    assert metrics["video_acc"] == 1.0
    assert metrics["video_f1"] == 1.0


def test_evaluate_video_metrics_mean_agg():
    model = _DummyModel()
    loader = [
        {
            "x": torch.zeros((2, 1, 3, 4, 4)),
            "y": torch.tensor([1.0, 0.0]),
            "video_id": ["v1", "v1"],
        }
    ]
    metrics = evaluate(
        model,
        loader,
        device=torch.device("cpu"),
        agg="mean",
        clip_threshold=0.5,
        video_threshold=0.5,
        ratio_threshold=0.5,
        use_video_metrics=True,
    )
    # Video metrics may be nan when only one class is present; clip metrics still valid
    assert metrics["clip_auc"] is not None


def test_evaluate_invalid_agg():
    model = _DummyModel()
    loader = [
        {
            "x": torch.zeros((2, 1, 3, 4, 4)),
            "y": torch.tensor([1.0, 0.0]),
            "video_id": ["v1", "v1"],
        }
    ]
    with pytest.raises(ValueError):
        evaluate(
            model,
            loader,
            device=torch.device("cpu"),
            agg="invalid",
            clip_threshold=0.5,
            video_threshold=0.5,
            ratio_threshold=0.5,
            use_video_metrics=True,
        )


def test_sample_indices_train_and_eval():
    import numpy as np

    rng = np.random.default_rng(0)
    idx_train = _sample_indices(t=2, k=4, train=True, rng=rng)
    assert len(idx_train) == 4
    assert all(i >= 0 for i in idx_train)
    # eval path is deterministic linspace
    idx_eval = _sample_indices(t=5, k=3, train=False, rng=np.random.default_rng(1))
    assert idx_eval.tolist() == [0, 2, 4]


def test_sample_indices_invalid_clip():
    import numpy as np
    with pytest.raises(ValueError):
        _sample_indices(t=0, k=1, train=True, rng=np.random.default_rng(0))


def test_summarize_profile_and_wandb_disabled(monkeypatch):
    records = [
        {"load_s": 0.1, "step_s": 0.2, "total_s": 0.3},
        {"load_s": 0.2, "step_s": 0.1, "total_s": 0.25},
    ]
    summary = _summarize_profile(records)
    assert summary is not None
    assert summary["samples"] == 2.0
    assert summary["load_max_s"] == 0.2

    # wandb disabled -> None
    from pathlib import Path

    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"wandb": {"enabled": False}})
    res = _maybe_init_wandb(cfg, Path("."))
    assert res is None
