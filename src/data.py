# src/data.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# -----------------------------
# Records
# -----------------------------
@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    rel_path: str          # relative to project root
    label: int             # 1 anomaly/accident, 0 normal
    subset: str            # positive/negative/challenging/rash/beanNG
    suffix: str
    size_bytes: int


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    video_id: str
    start_sec: float
    duration_sec: float
    target_fps: int
    n_frames: int
    img_size: int
    label: int
    subset: str
    split: str
    rel_path: str          # npz path relative to project root


# -----------------------------
# Video discovery
# -----------------------------
def _hash_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


def discover_videos(
    raw_dir: Path,
    include_challenging: bool = True,
    include_rash: bool = True,
    include_beamng: bool = True,
) -> list[VideoRecord]:
    """
    Discover videos under your described raw structure and assign labels/subsets.

    Assumptions (can be changed later):
      - Positive_Vidoes + challenging-environment + Rash-Driving (+ beanNG) => label=1
      - Negative_Videos => label=0
    """
    candidates: list[tuple[Path, str, int]] = []

    # Main folders
    pos_dir = raw_dir / "Final_videos" / "Positive_Vidoes"
    neg_dir = raw_dir / "Final_videos" / "Negative_Videos"
    chal_dir = raw_dir / "Final_videos" / "challenging-environment"
    rash_dir = raw_dir / "Rash-Driving"
    beamng_dir = rash_dir / "beanNG"

    if pos_dir.exists():
        candidates += [(p, "positive", 1) for p in pos_dir.glob("*.mov")]
    if neg_dir.exists():
        candidates += [(p, "negative", 0) for p in neg_dir.glob("*.mov")]

    if include_challenging and chal_dir.exists():
        candidates += [(p, "challenging", 1) for p in chal_dir.glob("*.mov")]

    if include_rash and rash_dir.exists():
        candidates += [(p, "rash", 1) for p in rash_dir.glob("*.mp4")]

    if include_rash and include_beamng and beamng_dir.exists():
        candidates += [(p, "beanNG", 1) for p in beamng_dir.glob("*.mp4")]

    if not candidates:
        raise FileNotFoundError(
            f"No videos found. Checked under: {raw_dir}. "
            f"Expected folders like data/raw/Final_videos/... and data/raw/Rash-Driving/..."
        )

    # De-dup (lightweight): same (name+suffix+size) considered same asset
    seen = {}
    records: list[VideoRecord] = []
    for path, subset, label in candidates:
        size = path.stat().st_size
        key = (path.name, path.suffix.lower(), size)
        if key in seen:
            # keep the "more anomalous" label if conflict (1 dominates 0)
            prev = seen[key]
            if label > prev.label:
                seen[key] = prev = VideoRecord(
                    video_id=prev.video_id,
                    rel_path=prev.rel_path,
                    label=label,
                    subset=prev.subset + f"+{subset}",
                    suffix=prev.suffix,
                    size_bytes=prev.size_bytes,
                )
            continue

        rel_path = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        video_id = _hash_id(rel_path, str(size), subset, str(label))
        rec = VideoRecord(
            video_id=video_id,
            rel_path=rel_path,
            label=label,
            subset=subset,
            suffix=path.suffix.lower(),
            size_bytes=size,
        )
        seen[key] = rec
        records.append(rec)

    # Sort for determinism
    records.sort(key=lambda r: (r.label, r.subset, r.rel_path))
    LOGGER.info("Discovered %d videos", len(records))
    return records


# -----------------------------
# Splitting (by video)
# -----------------------------
def split_videos(
    videos: list[VideoRecord],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, str]:
    """
    Returns mapping: video_id -> split in {train,val,test}
    Split is stratified by label (0/1) at video level.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[VideoRecord]] = {0: [], 1: []}
    for v in videos:
        by_label[v.label].append(v)

    mapping: dict[str, str] = {}
    for label, group in by_label.items():
        idx = np.arange(len(group))
        rng.shuffle(idx)
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train_ids = idx[:n_train]
        val_ids = idx[n_train: n_train + n_val]
        test_ids = idx[n_train + n_val:]

        for i in train_ids:
            mapping[group[i].video_id] = "train"
        for i in val_ids:
            mapping[group[i].video_id] = "val"
        for i in test_ids:
            mapping[group[i].video_id] = "test"

    return mapping


# -----------------------------
# Video decoding -> clips
# -----------------------------
def _read_clip_cv2(
    video_path: Path,
    start_sec: float,
    duration_sec: float,
    target_fps: int,
    img_size: int,
) -> np.ndarray:
    """
    Return uint8 RGB frames of shape [T, H, W, 3]
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "opencv-python is required for video decoding. "
            "Install with: pip install opencv-python"
        ) from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or np.isnan(src_fps) or src_fps <= 0:
        src_fps = 30.0  # fallback

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not total_frames or np.isnan(total_frames) or total_frames <= 0:
        total_frames = None
    else:
        total_frames = int(total_frames)

    start_frame = int(round(start_sec * src_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    step = max(int(round(src_fps / float(target_fps))), 1)
    n_frames = int(round(duration_sec * target_fps))
    frames: list[np.ndarray] = []

    # read enough frames to sample n_frames at 'step'
    read_budget = n_frames * step + step
    for i in range(read_budget):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if i % step != 0:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(
            frame_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb.astype(np.uint8))
        if len(frames) >= n_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            f"Got 0 frames from {video_path} at start={start_sec}")

    # pad if short
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    return np.stack(frames, axis=0)


def _safe_duration_seconds(video_path: Path) -> float:
    """
    Quick metadata read via cv2: duration = frame_count / fps
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "opencv-python is required. pip install opencv-python") from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0

    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if not n_frames or np.isnan(n_frames) or n_frames <= 0:
        # fallback guess
        return 0.0
    return float(n_frames) / float(fps)


def generate_clips_for_video(
    video: VideoRecord,
    video_split: str,
    clip_seconds: float,
    stride_seconds: float,
    target_fps: int,
    img_size: int,
    clips_dir: Path,
) -> list[ClipRecord]:
    video_path = PROJECT_ROOT / video.rel_path
    duration = _safe_duration_seconds(video_path)
    if duration <= 0:
        LOGGER.warning(
            "Could not read duration for %s, skipping.", video.rel_path)
        return []

    # start times (sliding window)
    max_start = max(0.0, duration - clip_seconds)
    starts = np.arange(0.0, max_start + 1e-6, stride_seconds)
    out: list[ClipRecord] = []

    for s in starts:
        clip_id = _hash_id(video.video_id, f"{s:.3f}", f"{clip_seconds:.3f}", str(
            target_fps), str(img_size))
        npz_rel = (
            clips_dir / f"{clip_id}.npz").resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        npz_path = PROJECT_ROOT / npz_rel

        if npz_path.exists():
            # already done
            frames = None
            n_frames = int(round(clip_seconds * target_fps))
        else:
            frames = _read_clip_cv2(
                video_path=video_path,
                start_sec=float(s),
                duration_sec=float(clip_seconds),
                target_fps=target_fps,
                img_size=img_size,
            )
            n_frames = int(frames.shape[0])
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_path,
                frames=frames,  # uint8 RGB
                video_id=video.video_id,
                start_sec=float(s),
                duration_sec=float(clip_seconds),
                target_fps=int(target_fps),
                img_size=int(img_size),
                label=int(video.label),
                subset=video.subset,
                source_rel_path=video.rel_path,
            )

        out.append(
            ClipRecord(
                clip_id=clip_id,
                video_id=video.video_id,
                start_sec=float(s),
                duration_sec=float(clip_seconds),
                target_fps=int(target_fps),
                n_frames=int(n_frames),
                img_size=int(img_size),
                label=int(video.label),
                subset=video.subset,
                split=video_split,
                rel_path=npz_rel,
            )
        )

    return out


# -----------------------------
# Writers
# -----------------------------
def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def preprocess(
    clip_seconds: float = 4.0,
    stride_seconds: float = 2.0,
    target_fps: int = 10,
    img_size: int = 224,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    include_challenging: bool = True,
    include_rash: bool = True,
    include_beamng: bool = True,
    force: bool = False,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifests_dir = PROCESSED_DIR / "manifests"
    clips_dir = PROCESSED_DIR / "clips"

    preprocess_cfg = {
        "clip_seconds": clip_seconds,
        "stride_seconds": stride_seconds,
        "target_fps": target_fps,
        "img_size": img_size,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "include_challenging": include_challenging,
        "include_rash": include_rash,
        "include_beamng": include_beamng,
    }

    cfg_path = manifests_dir / "preprocess.json"
    if cfg_path.exists() and not force:
        # If config matches, we can skip; else require force to avoid silent mismatch
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old == preprocess_cfg:
            LOGGER.info(
                "preprocess.json matches and force=False. Will reuse existing artifacts.")
        else:
            raise RuntimeError(
                "Existing preprocess.json differs from current config. "
                "Use --force to overwrite."
            )

    videos = discover_videos(
        raw_dir=RAW_DIR,
        include_challenging=include_challenging,
        include_rash=include_rash,
        include_beamng=include_beamng,
    )
    split_map = split_videos(
        videos, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)

    # Write videos manifest
    videos_csv = manifests_dir / "videos.csv"
    _write_csv(
        videos_csv,
        [asdict(v) | {"split": split_map.get(v.video_id, "train")}
         for v in videos],
    )

    # Generate clips
    all_clips: list[ClipRecord] = []
    for v in tqdm(videos, desc="Preprocessing videos -> clips"):
        v_split = split_map.get(v.video_id, "train")
        all_clips.extend(
            generate_clips_for_video(
                video=v,
                video_split=v_split,
                clip_seconds=clip_seconds,
                stride_seconds=stride_seconds,
                target_fps=target_fps,
                img_size=img_size,
                clips_dir=clips_dir,
            )
        )

    clips_csv = manifests_dir / "clips.csv"
    _write_csv(clips_csv, [asdict(c) for c in all_clips])

    manifests_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(preprocess_cfg, indent=2), encoding="utf-8")

    # Simple stats
    stats = {
        "n_videos": len(videos),
        "n_clips": len(all_clips),
        "by_split": {
            s: sum(1 for c in all_clips if c.split == s) for s in ["train", "val", "test"]
        },
        "by_label": {
            "0": sum(1 for c in all_clips if c.label == 0),
            "1": sum(1 for c in all_clips if c.label == 1),
        },
        "by_subset": {sub: sum(1 for c in all_clips if c.subset == sub) for sub in sorted({c.subset for c in all_clips})},
    }
    (manifests_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    LOGGER.info("Done. Wrote:\n- %s\n- %s\n- %s", videos_csv,
                clips_csv, manifests_dir / "stats.json")


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess TU-DAT videos into clip dataset.")
    p.add_argument("--clip-seconds", type=float, default=4.0)
    p.add_argument("--stride-seconds", type=float, default=2.0)
    p.add_argument("--target-fps", type=int, default=10)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--include-challenging", action="store_true", default=True)
    p.add_argument("--exclude-challenging", action="store_true", default=False)
    p.add_argument("--include-rash", action="store_true", default=True)
    p.add_argument("--exclude-rash", action="store_true", default=False)
    p.add_argument("--include-beamng", action="store_true", default=True)
    p.add_argument("--exclude-beamng", action="store_true", default=False)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(
    ), logging.INFO), format="%(levelname)s - %(message)s")

    preprocess(
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        target_fps=args.target_fps,
        img_size=args.img_size,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        include_challenging=(not args.exclude_challenging),
        include_rash=(not args.exclude_rash),
        include_beamng=(not args.exclude_beamng),
        force=args.force,
    )


if __name__ == "__main__":
    main()
