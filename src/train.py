from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.model import build_model

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_ROOT = ARTIFACTS_DIR / "checkpoints"
METRICS_ROOT = ARTIFACTS_DIR / "metrics"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFESTS_DIR = PROCESSED_DIR / "manifests"
CLIPS_CSV = MANIFESTS_DIR / "clips.csv"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _sample_indices(
    t: int, k: int, train: bool, rng: np.random.Generator
) -> np.ndarray:
    if t <= 0:
        raise ValueError("Clip has 0 frames.")
    if train:
        replace = t < k
        idx = rng.choice(t, size=k, replace=replace)
        idx.sort()
        return idx
    return np.linspace(0, t - 1, num=k).round().astype(int)


class ClipDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, k_frames: int, train: bool, seed: int = 42
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.k_frames = int(k_frames)
        self.train = bool(train)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        npz_path = PROJECT_ROOT / row["rel_path"]
        data = np.load(npz_path, allow_pickle=False)
        frames = data["frames"]
        label = int(row["label"])
        video_id = str(row["video_id"])

        t = int(frames.shape[0])
        sel = _sample_indices(t, self.k_frames, self.train, self.rng)
        frames = frames[sel]

        x = torch.from_numpy(frames).float() / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()

        mean = IMAGENET_MEAN.to(dtype=x.dtype)
        std = IMAGENET_STD.to(dtype=x.dtype)
        x = (x - mean) / std

        y = torch.tensor(label, dtype=torch.float32)
        return {"x": x, "y": y, "video_id": video_id}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    agg: str,
    clip_threshold: float,
    video_threshold: float,
    ratio_threshold: float,
    use_video_metrics: bool = True,
) -> dict[str, Any]:
    model.eval()
    y_true, y_prob = [], []
    video_probs: dict[str, list[float]] = {} if use_video_metrics else {}
    video_label: dict[str, int] = {} if use_video_metrics else {}

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy().astype(int).tolist()
        vids = batch["video_id"]

        logits = model(x)
        prob = torch.sigmoid(logits).detach().cpu().numpy().tolist()

        y_true.extend(y)
        y_prob.extend(prob)

        if use_video_metrics:
            for v, p, yy in zip(vids, prob, y):
                video_probs.setdefault(v, []).append(float(p))
                video_label.setdefault(v, int(yy))

    out: dict[str, Any] = {}
    out["clip_acc"] = float(
        accuracy_score(y_true, [int(p >= clip_threshold) for p in y_prob])
    )
    out["clip_f1"] = float(
        f1_score(y_true, [int(p >= clip_threshold)
                 for p in y_prob], zero_division=0)
    )
    try:
        out["clip_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["clip_auc"] = None

    if not use_video_metrics:
        out["video_acc"] = None
        out["video_f1"] = None
        out["video_auc"] = None
        return out

    v_true, v_prob = [], []
    for v, probs in video_probs.items():
        if agg == "max":
            vp = max(probs)
        elif agg == "mean":
            vp = float(np.mean(probs))
        elif agg == "ratio":
            vp = float(np.mean([p >= ratio_threshold for p in probs]))
        else:
            raise ValueError("agg must be max|mean|ratio")
        v_prob.append(vp)
        v_true.append(video_label[v])

    out["video_acc"] = float(
        accuracy_score(v_true, [int(p >= video_threshold) for p in v_prob])
    )
    out["video_f1"] = float(
        f1_score(v_true, [int(p >= video_threshold)
                 for p in v_prob], zero_division=0)
    )
    try:
        out["video_auc"] = float(roc_auc_score(v_true, v_prob))
    except Exception:
        out["video_auc"] = None

    return out


def _maybe_init_wandb(cfg: DictConfig, reports_dir: Path) -> Optional[Any]:
    """Initialize wandb run if enabled in config."""
    wandb_cfg = cfg.get("wandb")
    if not wandb_cfg or not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        LOGGER.warning("wandb not installed; skipping wandb logging.")
        return None

    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name") or reports_dir.name,
        tags=wandb_cfg.get("tags") or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(reports_dir),
        mode=wandb_cfg.get("mode"),
        reinit=True,
    )
    LOGGER.info("Initialized wandb run: %s", run.name)
    return run


def train(cfg: DictConfig) -> None:
    if not CLIPS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {CLIPS_CSV}. Run preprocessing first.")

    df = pd.read_csv(CLIPS_CSV)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not cfg.train.cpu else "cpu"
    )
    LOGGER.info("Device: %s", device)

    train_ds = ClipDataset(
        train_df, k_frames=cfg.train.k_frames, train=True, seed=cfg.train.seed
    )
    val_ds = ClipDataset(
        val_df, k_frames=cfg.train.k_frames, train=False, seed=cfg.train.seed
    )
    test_ds = ClipDataset(
        test_df, k_frames=cfg.train.k_frames, train=False, seed=cfg.train.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    model = build_model(
        cfg.model.name, pretrained=cfg.model.pretrained).to(device)

    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())
    pos_weight = None
    if n_pos > 0:
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32, device=device
        )
        LOGGER.info(
            "pos_weight = %.4f (neg=%d, pos=%d)", float(
                pos_weight.item()), n_neg, n_pos
        )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd
    )

    best_score = -1.0
    reports_dir = Path(HydraConfig.get().runtime.output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_suffix = reports_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        run_suffix = Path(reports_dir.name)
    ckpt_dir = CHECKPOINTS_ROOT / run_suffix
    metrics_dir = METRICS_ROOT / run_suffix
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best.pt"
    wandb_run = _maybe_init_wandb(cfg, reports_dir)
    wandb_cfg = cfg.get("wandb") or {}

    history: list[dict[str, Any]] = []

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs}"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            agg=cfg.eval.agg,
            clip_threshold=cfg.eval.clip_threshold,
            video_threshold=cfg.eval.video_threshold,
            ratio_threshold=cfg.eval.ratio_threshold,
            use_video_metrics=cfg.eval.use_video_metrics,
        )

        row = {"epoch": epoch, "train_loss": train_loss} | val_metrics
        history.append(row)
        LOGGER.info(
            "Epoch %d | loss=%.4f | val clip_auc=%s clip_f1=%.3f video_auc=%s",
            epoch,
            train_loss,
            str(val_metrics.get("clip_auc")),
            val_metrics.get("clip_f1", 0.0),
            str(val_metrics.get("video_auc")),
        )

        if wandb_run:
            log_payload = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val/{k}": v for k, v in val_metrics.items() if v is not None},
            }
            wandb_run.log(log_payload, step=epoch)

        score_candidates = [
            val_metrics.get("video_auc") if cfg.eval.use_video_metrics else None,
            val_metrics.get("clip_auc"),
            val_metrics.get("clip_f1"),
        ]
        score = next((s for s in score_candidates if s is not None), 0.0)

        if float(score) > best_score:
            best_score = float(score)
            ckpt = {
                "model": cfg.model.name,
                "state_dict": model.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "best_score": best_score,
                "epoch": epoch,
            }
            torch.save(ckpt, best_ckpt_path)
            LOGGER.info(
                "Saved best checkpoint (score=%.4f) to %s", best_score, best_ckpt_path
            )
            if wandb_run and wandb_cfg.get("log_checkpoints", False):
                import wandb  # type: ignore

                artifact = wandb.Artifact(
                    name=f"{cfg.model.name}-best",
                    type="model",
                    metadata={"best_score": best_score, "epoch": epoch},
                )
                artifact.add_file(str(best_ckpt_path))
                wandb_run.log_artifact(artifact)

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    test_metrics = evaluate(
        model,
        test_loader,
        device=device,
        agg=cfg.eval.agg,
        clip_threshold=cfg.eval.clip_threshold,
        video_threshold=cfg.eval.video_threshold,
        ratio_threshold=cfg.eval.ratio_threshold,
        use_video_metrics=cfg.eval.use_video_metrics,
    )
    out = {
        "best_val_score": best_score,
        "test": test_metrics,
        "history": history,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    metrics_path = metrics_dir / "metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", metrics_path)
    # Keep a copy in the Hydra run directory for convenience.
    hydra_metrics_path = reports_dir / "metrics.json"
    if hydra_metrics_path != metrics_path:
        hydra_metrics_path.write_text(
            json.dumps(out, indent=2), encoding="utf-8")
        LOGGER.info("Wrote %s", hydra_metrics_path)

    if wandb_run:
        wandb_run.summary["best_val_score"] = best_score
        wandb_run.log(
            {f"test/{k}": v for k, v in test_metrics.items() if v is not None},
            step=cfg.train.epochs + 1,
        )
        if wandb_cfg.get("log_artifacts", False):
            import wandb  # type: ignore

            metrics_artifact = wandb.Artifact(
                name=f"metrics-{reports_dir.name}",
                type="metrics",
            )
            metrics_artifact.add_file(str(metrics_path))
            if hydra_metrics_path.exists() and hydra_metrics_path != metrics_path:
                metrics_artifact.add_file(str(hydra_metrics_path))
            wandb_run.log_artifact(metrics_artifact)
        wandb_run.finish()


@hydra.main(config_path="../configs/train", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.train.log_level.upper(), logging.INFO),
        format="%(levelname)s - %(message)s",
    )
    train(cfg)


if __name__ == "__main__":
    main()
