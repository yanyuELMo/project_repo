from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import VideoRecord, split_videos
import src.train as train


def test_split_videos_covers_all_and_keeps_labels():
    videos: list[VideoRecord] = []
    for i in range(4):
        videos.append(
            VideoRecord(
                video_id=f"pos{i}",
                rel_path=f"pos{i}.mp4",
                label=1,
                subset="pos",
                suffix=".mp4",
                size_bytes=100 + i,
            )
        )
    for i in range(4):
        videos.append(
            VideoRecord(
                video_id=f"neg{i}",
                rel_path=f"neg{i}.mp4",
                label=0,
                subset="neg",
                suffix=".mp4",
                size_bytes=200 + i,
            )
        )

    mapping = split_videos(videos, seed=0, train_ratio=0.5, val_ratio=0.25)

    assert set(mapping.keys()) == {v.video_id for v in videos}
    counts = {s: list(mapping.values()).count(s) for s in ("train", "val", "test")}
    assert counts["train"] > 0 and counts["val"] > 0
    # ensure both labels appear in train split (stratified)
    train_labels = {v.label for v in videos if mapping[v.video_id] == "train"}
    assert train_labels == {0, 1}


def test_clip_dataset_returns_normalized_tensor(tmp_path, monkeypatch):
    # isolate artifacts to the pytest temp dir by patching PROJECT_ROOT
    monkeypatch.setattr(train, "PROJECT_ROOT", Path(tmp_path))

    frames = np.random.randint(0, 255, size=(5, 8, 8, 3), dtype=np.uint8)
    clip_path = Path(tmp_path) / "clip.npz"
    np.savez(clip_path, frames=frames)

    df = pd.DataFrame(
        [
            {
                "rel_path": "clip.npz",
                "label": 1,
                "video_id": "vid1",
                "split": "train",
            }
        ]
    )

    ds = train.ClipDataset(df, k_frames=3, train=False, seed=0)
    sample = ds[0]

    assert sample["x"].shape == (3, 3, 8, 8)
    assert sample["x"].dtype == torch.float32
    assert sample["y"].item() == 1
    assert torch.isfinite(sample["x"]).all()


def test_discover_videos_basic(tmp_path):
    root = tmp_path / "data" / "raw"
    pos_dir = root / "Final_videos" / "Positive_Vidoes"
    neg_dir = root / "Final_videos" / "Negative_Videos"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)
    (pos_dir / "p1.mov").write_bytes(b"0")
    (neg_dir / "n1.mov").write_bytes(b"0")

    # Patch PROJECT_ROOT so discover_videos treats tmp_path as project root
    from src import data as data_mod
    data_mod.PROJECT_ROOT = tmp_path
    videos = data_mod.discover_videos(root, include_challenging=False, include_rash=False, include_beamng=False)
    labels = {v.video_id: v.label for v in videos}
    subsets = {v.video_id: v.subset for v in videos}

    assert len(videos) == 2
    assert set(labels.values()) == {0, 1}
    assert set(subsets.values()) == {"positive", "negative"}
