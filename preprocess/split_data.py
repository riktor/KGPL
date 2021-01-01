import numpy as np
import pickle
import hydra


def load_rating(cfg):
    print("reading rating file ...")

    # reading rating file
    all_rating_np = np.load(hydra.utils.to_absolute_path(cfg.rating_path))
    rating_np, test_data = split_data(all_rating_np)
    rating_np = rating_np[rating_np[:, 2] == 1]
    train_data, eval_data = split_data(rating_np)

    n_user = len(set(all_rating_np[:, 0]))
    n_item = len(set(all_rating_np[:, 1]))

    return (
        n_user,
        n_item,
        train_data,
        eval_data,
        test_data,
    )


def split_data(rating_np, split_ratio=0.2):
    n_ratings = rating_np.shape[0]
    all_indices = list(range(n_ratings))
    split_indices = np.random.choice(all_indices, size=int(n_ratings * split_ratio), replace=False)
    splitted_data = rating_np[split_indices]
    rest_data = rating_np[~np.isin(all_indices, split_indices)]
    return rest_data, splitted_data


@hydra.main(config_path="../conf/config.yaml")
def main(cfg):
    print(cfg.pretty())

    data = load_rating(cfg)

    print("data loaded.")
    pickle.dump(data, open(hydra.utils.to_absolute_path(cfg.data_path), "wb"))


if __name__ == "__main__":
    main()
