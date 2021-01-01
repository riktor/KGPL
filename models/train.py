import time
import pickle
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import pandas as pd
import hydra
from loguru import logger
from prettytable import PrettyTable
import multiprocessing as mp
from pathlib import Path

from kgpl import KGPL_COT
from utils import grouper

from tqdm import tqdm

tf.set_random_seed(2021)
np.random.seed(2021)


def train(cfg, data):
    (n_user, n_item, train_data, eval_data, test_data) = data

    adj_entity = np.load(hydra.utils.to_absolute_path(cfg.adj_entity_path))
    adj_relation = np.load(hydra.utils.to_absolute_path(cfg.adj_relation_path))
    n_entity = adj_entity.shape[0]
    n_relation = len(np.unique(np.reshape(adj_relation, -1)))

    logger.info(f"num train records: {len(train_data)}")

    logger.info(f"num adj entities: {len(adj_entity)}, num entities: {n_entity}")
    logger.info(f"num adj relations: {len(adj_relation)}, num relations: {n_relation}")

    path_list_dict = pickle.load(open(hydra.utils.to_absolute_path(cfg.pathlist_path), "rb"))
    
    model = KGPL_COT(
        cfg,
        n_user,
        n_item,
        n_entity,
        n_relation,
        adj_entity,
        adj_relation,
        path_list_dict,
        train_data,
        eval_data,
    )
    _pos_inds = train_data[:, 2] == 1
    train_data = train_data[_pos_inds]
    print("model type:", cfg.model_type)

    # top-K evaluation settings
    topk_config = topk_settings(
        train_data,
        eval_data,
        test_data,
        n_item,
        test_mode=True,
        user_num=cfg.evaluate.user_num_topk,
    )
    batch_size = cfg.optimize.batch_size

    saver = tf.train.Saver()
    ckpt_path = Path(hydra.utils.to_absolute_path(cfg.log.ckpt_path))
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        logger.info(f"Session start!")
        sess.run(tf.global_variables_initializer())
        sess.run(model.f.zero_ops)
        sess.run(model.g.zero_ops)

        train_data_pl = None
        for step in range(cfg.optimize.n_epochs):
            # training
            for i in range(cfg.optimize.iter_per_epoch):
                np.random.shuffle(train_data)
                start = 0
                while start + batch_size <= train_data.shape[0]:
                    fd = model.get_feed_dict(
                        train_data,
                        start,
                        start + batch_size,
                        epoch=step,
                        sess=sess,
                        eval_mode=False,
                    )
                    metrics = model.train(sess, fd)
                    if cfg.log.show_loss:
                        fmetrics = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                        logger.info(f"{start} total loss: {fmetrics}")
                    start += batch_size

            pr_table, precision, recall = evaluate(
                sess,
                model,
                n_item,
                train_data,
                eval_data,
                test_data,
                test_mode=False,
                user_num_topk=cfg.evaluate.user_num_topk,
            )
            logger.info(f"epoch {step}  P/R\n {pr_table}")
            k_list = topk_config[-1]

            ts = time.time()
            path_name = f'ckpt_{cfg.log.experiment_name}_{step}_{ts}'
            export_path = ckpt_path / f"{path_name}/{path_name}"
            saver.save(sess, str(export_path))

        pr_table, precision, recall = evaluate(
            sess,
            model,
            n_item,
            train_data,
            eval_data,
            test_data,
            test_mode=True,
            user_num_topk=cfg.evaluate.user_num_topk,
        )
        logger.info(f"epoch {step} test  P/R\n {pr_table}")


def topk_settings(
    train_data,
    eval_data,
    test_data,
    n_item,
    user_num=500,
    test_mode=False,
    k_list=[1, 2, 5, 10, 20, 50, 100],
):
    if test_mode:
        train_record = get_user_record(np.vstack([train_data, eval_data]), True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_data[:, 0]) & set(test_record.keys()))
    else:
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(eval_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))

    if user_num is None:
        user_num = len(user_list)

    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))
    return user_list, train_record, test_record, item_set, k_list


def get_feed_dict(model, data, start, end):
    feed_dict = {
        model.user_indices: data[start:end, 0],
        model.item_indices: data[start:end, 1],
        model.labels: data[start:end, 2],
    }
    return feed_dict


def evaluate(
    sess, model, n_item, train_data, eval_data, test_data, test_mode=False, user_num_topk=400
):
    user_list, train_record, test_record, item_set, k_list = topk_settings(
        train_data, eval_data, test_data, n_item, test_mode=test_mode, user_num=user_num_topk
    )

    # Top-K
    precision, recall, _ = topk_eval(sess, model,
                                     user_list, train_record, test_record,
                                     item_set, k_list, model.cfg.optimize.batch_size)
    pr_table = PrettyTable(["measure"] + k_list)
    pr_table.add_row(["precision"] + [f"{n:.4f}" for n in precision])
    pr_table.add_row(["recall"] + [f"{n:.4f}" for n in recall])
    return pr_table, precision, recall


def topk_eval(
    sess,
    model,
    user_list,
    train_record,
    test_record,
    item_set,
    k_list,
    batch_size,
    chunk_size=30,
    average_mode=True,
    par=16,
    return_topk=False,
):
    """Compute ranking measures 
    Args:
        sess: TF session.
        model: TF model,
        user_list: list of target users
        train_record: dict of training data provided by `get_ser_record`
        test_record: dict of testing data provided by `get_ser_record`
        item_set: set of all items
        k_list: list of K's for evaluation, such as 10, 20, 50
        batch_size: batch size for score inference
        chunk_size: chunk size for parallel computation
        par: number of processes for parallel computation
        return_topk: True/False. When setting True, this function returns the top-K ranked lists. 
    Returns:
        lists of precision, recall, and (optionally) ranked lists.
    """
    users, items = [], []
    user_inds_dict = {}
    s = 0
    for u in tqdm(user_list):
        if u not in train_record:
            continue
        i_s = tuple(item_set - train_record[u])
        u_s = [u] * len(i_s)
        users.extend(u_s)
        items.extend(i_s)
        user_inds_dict[u] = (s, s + len(u_s))
        s += len(u_s)

    # create rankings
    scores = []
    for start in tqdm(range(0, len(users), batch_size)):
        end = start + batch_size
        _users = users[start:end]
        _items = items[start:end]
        n = len(_users)
        if n < batch_size:
            _users = _users + [_users[0]] * (batch_size - n)
            _items = _items + [_items[0]] * (batch_size - n)
        try:
            _, _scores = model.get_scores(sess, _users, _items)
        except:
            _scores = [0] * len(_users)
        scores.append(_scores[:n])
        start += batch_size
    scores = np.concatenate(scores)

    # split the data to evaluate
    src_itr = [
        (u, items[s:e], scores[s:e], test_record.get(u, set()), k_list, return_topk)
        for u, (s, e) in user_inds_dict.items()
    ]
    grouped = grouper(chunk_size, src_itr, squash=set([4]))

    # comupte measures for the splitted data
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ranking_list = []
    with mp.Pool(par) as pool:
        for precision_k, recall_k, rk_ls in tqdm(
            pool.imap_unordered(compute_rank_measures_, grouped),
            total=len(user_inds_dict) // chunk_size,
        ):
            for k in k_list:
                precision_list[k].extend(precision_k[k])
                recall_list[k].extend(recall_k[k])
            ranking_list.extend(rk_ls)
    if average_mode:
        precision = [np.mean(precision_list[k]) for k in k_list]
        recall = [np.mean(recall_list[k]) for k in k_list]
    else:
        precision = [precision_list[k] for k in k_list]
        recall = [recall_list[k] for k in k_list]
    return precision, recall, ranking_list


def compute_rank_measures_(args_list):
    """
    The function for parallel computation of ranking measures.
    """
    _, _, _, _, k_list, return_topk = args_list[0]
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ranking_list = []
    for args in args_list:
        if args is None:
            continue
        u, _items, _scores, test_record, _, _ = args

        item_score_map = {}
        for item, score in zip(_items, _scores):
            item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        if return_topk:
            ranking_list.append([(i, i in test_record) for i in item_sorted[: k_list[-1]]])

        N = len(item_sorted)
        for k in k_list:
            if len(test_record) == 0:
                continue
            hit_num = len(set(item_sorted[:k]) & test_record)
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record))
    return precision_list, recall_list, ranking_list


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
