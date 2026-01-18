import hydra
from omegaconf import DictConfig
from src.data import preprocess


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    preprocess(**cfg.data)


if __name__ == "__main__":
    main()
