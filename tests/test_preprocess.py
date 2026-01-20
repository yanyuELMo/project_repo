import json

from src.data import ClipRecord, preprocess


def test_preprocess_writes_manifests(tmp_path, monkeypatch):
    # minimal setup: create raw dir structure with one dummy video file
    raw_root = tmp_path / "data" / "raw" / "Rash-Driving"
    raw_root.mkdir(parents=True, exist_ok=True)
    dummy_video = raw_root / "vid.mp4"
    dummy_video.write_bytes(b"fakevideo")

    monkeypatch.setattr("src.data.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.data.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("src.data.RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr("src.data.PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr("src.data._safe_duration_seconds", lambda _p: 1.0)

    def _fake_clips(*args, **kwargs):
        return [
            ClipRecord(
                clip_id="c1",
                video_id="v1",
                start_sec=0.0,
                duration_sec=1.0,
                target_fps=1,
                n_frames=1,
                img_size=8,
                label=1,
                subset="rash",
                split="train",
                rel_path="processed/clips/c1.npz",
            )
        ]

    monkeypatch.setattr("src.data.generate_clips_for_video", _fake_clips)

    preprocess(
        clip_seconds=1.0,
        stride_seconds=1.0,
        target_fps=1,
        img_size=8,
        seed=0,
        train_ratio=0.8,
        val_ratio=0.1,
        include_challenging=False,
        include_rash=True,
        include_beamng=False,
        force=True,
    )

    manifests = tmp_path / "data" / "processed" / "manifests"
    assert (manifests / "videos.csv").exists()
    assert (manifests / "clips.csv").exists()
    stats = json.loads((manifests / "stats.json").read_text())
    assert stats["n_videos"] == 1
