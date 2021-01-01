import numpy as np
import pickle
import joblib
import hydra
from collections import defaultdict
from pathlib import Path
from operator import itemgetter

global kg
global adj_entity
global adj_relation
global depth


def prepare_kg(kg_path):
    global kg

    kg_np = np.load(kg_path)

    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def construct_adj(kg):
    global adj_entity
    global adj_relation

    max_len = np.max([len(kg[e]) for e in kg])
    adj_entity = []
    adj_relation = []
    for entity in kg:
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        sampled_indices = list(range(n_neighbors)) + [-1] * max(max_len - n_neighbors, 0)
        adj_entity.append(np.array([neighbors[i][0] for i in sampled_indices]))
        adj_relation.append(np.array([neighbors[i][1] for i in sampled_indices]))
    adj_entity = np.array(adj_entity, dtype=np.int64)
    adj_relation = np.array(adj_relation, dtype=np.int64)

    return adj_entity, adj_relation


def construct_adj_random(kg, num_neighbor_samples):
    global adj_entity
    global adj_relation

    n_entity = np.max(list(kg.keys())) + 1
    adj_entity = np.zeros((n_entity, num_neighbor_samples), dtype=np.int64)
    adj_relation = np.zeros((n_entity, num_neighbor_samples), dtype=np.int64)
    for entity in kg:
        entity = int(entity)
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= num_neighbor_samples:
            sampled_indices = np.random.choice(
                list(range(n_neighbors)), size=num_neighbor_samples, replace=False
            )
        else:
            sampled_indices = np.random.choice(
                list(range(n_neighbors)), size=num_neighbor_samples, replace=True
            )
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])
    return adj_entity, adj_relation


def get_all_items(rating_path):
    rating_np = np.load(rating_path)
    all_items = list(set(rating_np[:, 1]))
    return all_items


def get_paths(seed_item):
    # path finding based on BFS
    accepted_neighbors = defaultdict(set)
    paths = set()
    queue = [[seed_item]]
    while len(queue) > 0:
        pt = queue.pop(0)
        e = pt[-1]
        next_e = list(set(adj_entity[e]))
        for ne in next_e:
            if ne == -1 and (len(pt) > 1 and ne == pt[-2]):
                continue
            next_pt = pt[::] + [ne]
            if ne in all_items:
                paths.add(tuple(next_pt))
                for s, e in zip(next_pt[:-1], next_pt[0:]):
                    accepted_neighbors[s].add(e)
                    accepted_neighbors[e].add(s)
            if len(next_pt) < depth:
                queue.append(next_pt)
    return paths, accepted_neighbors


@hydra.main(config_path="../conf/preprocess.yaml")
def main(cfg):
    global depth
    global all_items

    depth = cfg.lp_depth
    print(cfg.pretty())

    rating_path = hydra.utils.to_absolute_path(cfg.rating_path)
    all_items = get_all_items(rating_path)

    kg_path = hydra.utils.to_absolute_path(cfg.kg_path)
    kg = prepare_kg(kg_path)

    # contruct adjecency matrix 
    adj_entity, adj_relation = construct_adj_random(kg, cfg.num_neighbor_samples)

    # path finding based on BFS
    results = joblib.Parallel(n_jobs=32, verbose=10, backend="multiprocessing")(
        [joblib.delayed(get_paths)(i) for i in all_items]
    )
    path_set_list = list(map(itemgetter(0), results))

    save_dir = Path("data") / cfg.dataset
    pl_save_path = hydra.utils.to_absolute_path(
        str(
            save_dir / f"path_list_{cfg.lp_depth}_{cfg.num_neighbor_samples}.pkl"
        )
    )
    adje_save_path = hydra.utils.to_absolute_path(
        str(save_dir / f"adj_entity_{cfg.lp_depth}_{cfg.num_neighbor_samples}")
    )
    adjr_save_path = hydra.utils.to_absolute_path(
        str(save_dir / f"adj_relation_{cfg.lp_depth}_{cfg.num_neighbor_samples}")
    )

    lens = []
    for ps in path_set_list:
        lens.append(len(ps))
    print("average number of paths:", np.average(lens))
    print("median number of paths:", np.median(lens))
    print("min number of paths:", np.min(lens))
    print("max number of paths:", np.max(lens))

    pickle.dump(dict(zip(all_items, path_set_list)), open(pl_save_path, "wb"))
    np.save(adje_save_path, adj_entity)
    np.save(adjr_save_path, adj_relation)


if __name__ == "__main__":
    main()
