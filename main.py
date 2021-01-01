import sys
import hydra
import pickle

sys.path.append("models/")
from train import train


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    print(cfg.pretty())
    print(cfg.log.experiment_name)

    data = pickle.load(open(hydra.utils.to_absolute_path(cfg.data_path), "rb"))
    train(cfg, data)


if __name__ == "__main__":
    main()
