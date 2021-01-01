import random
import itertools
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.metrics import f1_score, roc_auc_score
from collections import defaultdict, Counter
import hydra
import multiprocessing as mp
from loguru import logger
import copy

from aggregators import SumAggregatorWithDropout
from utils import grouper

tf.set_random_seed(2021)
np.random.seed(2021)
random.seed(2021)


from tqdm import tqdm


def compute_reachable_items_(args_list):
    """Construct the sampling distributions based on paths in KG.
    Args:
        args_list: list of list of arguments. Each arguments' list must contains;
        (1) user_id;
        (2) user's interacted item ids (seed items);
        (3) item-to-(item, #paths) dict found in the BFS (start and end points of some paths);
        (4) item-to-frequency dict;
        (5) power coefficient to control the skewness of sampling distributions
    Returns:
        dict in which (key, value) = (item list, np.array of sampling distribution).
        sampling distribution is transformed to CDF for fast sampling. 
    """
    idd = {}
    _, _, dst_dict, item_freq, pn = args_list[0]
    for args in args_list:
        if args is None:
            continue
        user, seed_items, _, _, _ = args

        # Collect user's reachable items with the number of reachable paths
        dst = Counter()
        for item in seed_items:
            if item in dst_dict:
                dst += dst_dict[item]
        
        if len(dst) != 0:
            # Unique reachable items for the user
            udst = np.array(tuple(dst.keys()))
            
            # Histogram of paths with power transform
            F = np.array(tuple(dst.values())) ** pn
            
            # Remove the seed (positve) items 
            inds = ~np.isin(udst, seed_items)
            udst = udst[inds]            
            F = F[inds]

            # Compute unreachable items and concat those to the end of item lists
            udst = set(udst)
            unreachable_items = [i for i in item_freq if i not in udst]
            udst = list(udst) + unreachable_items

            # For unreachable items, assume 0.5 virtual paths for full support
            F = np.concatenate([F, np.ones(len(unreachable_items)) * 0.5])
            
            # Transform histogram to CDF
            sort_inds = np.argsort(F)
            udst = [udst[i] for i in sort_inds]
            F = F[sort_inds]
            F = (F / np.sum(F)).cumsum()
            idd[user] = (udst, F)
    return idd


def compute_user_unseen_items_(args_list):
    user_unseen_items = {}
    all_items, _, _ = args_list[0]
    for args in args_list:
        if args is None:
            continue
        _, user, seed_items, _, _ = args
        unseen_items = tuple(all_items - seed_items)
        user_unseen_items[user] = (unseen_items, None)
    return user_unseen_items


class KaPLMixin(object):
    def __init__(
        self, cfg, n_entity, n_relation, adj_entity, adj_relation, path_list_dict, eval_mode=False
    ):
        self.user_seed_dict = defaultdict(set)
        self.item_dist_dict = {}
        self.cand_uinds = None
        self.cand_iinds = None
        self.n_entity = n_entity
        self.n_relation = n_relation

    def _build_freq_dict(self, seq, all_candidates):        
        _freq = Counter(seq)
        for i in all_candidates:
            if i not in _freq:
                _freq[i] += 1
        freq = [_freq[i] for i in all_candidates]
        return dict(zip(all_candidates, freq))

    def set_item_candidates(
        self, n_user, n_item, train_data, eval_data, path_list_dict
    ):
        """Construct the sampling distrbiutions for negative/pseudo-labelled instances for each user
        """
        all_users = tuple(set(train_data[:, 0]))
        self.all_users = all_users
        
        self.n_item = n_item
        self.all_items = set(range(n_item))
        self.neg_c_dict_user = self._build_freq_dict(
            np.concatenate([train_data[:, 0], eval_data[:, 0]]), self.all_users
        )
        self.neg_c_dict_item = self._build_freq_dict(
            np.concatenate([train_data[:, 1], eval_data[:, 1]]), self.all_items
        )

        item_cands = tuple(self.neg_c_dict_item.keys())
        F = np.array(tuple(self.neg_c_dict_item.values())) ** self.cfg.plabel.neg_pn
        sort_inds = np.argsort(F)
        item_cands = [item_cands[i] for i in sort_inds]
        F = F[sort_inds]
        F = (F / F.sum()).cumsum()
        self.item_freq = (item_cands, F)

        for u, i in tqdm(train_data[:, 0:2]):
            self.user_seed_dict[u].add(i)

        path = hydra.utils.to_absolute_path(self.cfg.reachable_items_path)
        logger.info("calculating reachable items for users")
        self._setup_dst_dict(path_list_dict)
        item_dist_dict = {}
        src_itr = map(
            lambda iu: (
                all_users[iu],
                tuple(self.user_seed_dict[all_users[iu]]),
                self.dst_dict,
                self.neg_c_dict_item,
                self.cfg.plabel.pl_pn,
            ),
            range(len(all_users)),
        )
        grouped = grouper(self.cfg.plabel.chunk_size, src_itr, squash=set([2, 3]))
        with mp.Pool(self.cfg.plabel.par) as pool:
            for idd in pool.imap_unordered(compute_reachable_items_, grouped):
                item_dist_dict.update(idd)
        self.item_dist_dict = item_dist_dict

    def _setup_dst_dict(self, path_list_dict):
        """
        Transform path representations:
        `list of nodes` to `dictionaly of source to sink (dst_dict)`
        """

        logger.info("setup dst dict...")
        dst_dict = {}
        for item in tqdm(path_list_dict):
            dst = []
            paths = path_list_dict[item]
            for i, p in enumerate(paths):
                dst.append(p[-1])
            dst_dict[item] = Counter(dst)
        logger.info("start updating path info...")
        self.dst_dict = dst_dict
        logger.info("path info updated.")

    def _get_user_rel_scores(self, sess, users):
        uembs = sess.run(
            self.user_embeddings, feed_dict={self.user_indices: users, self.dropout_rate: 0.0}
        )  # nu, legth
        rembs = sess.run(self.relation_emb_matrix)  # nr, length

        return np.dot(uembs, rembs.T)  # nu, nr

    def _get_mini_batch_pl(self, sess, users):
        """
        Create pseudo-labelled instances for users
        """
        pl_users, pl_items = [], []
        ind = 0
        cands, freq_F = self.item_freq
        while True:
            u = users[ind % len(users)]
            ind += 1
            if u in self.item_dist_dict and len(self.item_dist_dict[u][0]) != 0:
                udst, F = self.item_dist_dict[u]
                i = udst[np.searchsorted(F, random.random())]
            else:
                while True:
                    i = cands[np.searchsorted(freq_F, random.random())]
                    if i not in self.user_seed_dict[u]:
                        break
            pl_users.append(u)
            pl_items.append(i)
            if len(pl_users) == len(users):
                break

        pl_users_pad = list(pl_users) + [0] * (self.batch_size - len(pl_users))
        pl_items_pad = list(pl_items) + [0] * (self.batch_size - len(pl_items))
        pl_labels_pad = sess.run(
            self.scores_normalized,
            feed_dict={
                self.user_indices: pl_users_pad,
                self.item_indices: pl_items_pad,
                self.dropout_rate: 0.0,
            },
        )
        pl_users = pl_users_pad[: len(pl_users)]
        pl_items = pl_items_pad[: len(pl_items)]
        pl_labels = pl_labels_pad[: len(pl_users)]
        return pl_users, pl_items, pl_labels


class KGPL_STUDENT(KaPLMixin):
    def __init__(
        self,
        cfg,
        n_user, n_entity, n_relation,
        adj_entity, adj_relation, path_list_dict,
        name, eval_mode=False,
    ):
        self.n_user = n_user
        self.name = name
        self._parse_cfg(cfg, adj_entity, adj_relation)
        KaPLMixin.__init__(
            self,
            cfg,
            n_entity,
            n_relation,
            adj_entity,
            adj_relation,
            path_list_dict,
            eval_mode=eval_mode,
        )

        with tf.variable_scope(self.name) as scope:
            self._build_inputs()
            self._build_model(n_user, n_entity, n_relation)
            self._build_train(scope)

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_cfg(self, cfg, adj_entity, adj_relation):
        self.cfg = cfg
        self.batch_size = cfg.optimize.batch_size

        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name="user_indices")
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name="item_indices")
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")
        self.neg_mask = tf.placeholder(dtype=tf.float32, shape=[None], name="negative_mask")
        self.plabel_mask = tf.placeholder(dtype=tf.float32, shape=[None], name="plabel_mask")
        self.dropout_rate = tf.placeholder(dtype=tf.float32, name="dropout_rate")

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.cfg.emb_dim],
            initializer=KGPL_STUDENT.get_initializer(),
            name="user_emb_matrix",
        )
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.cfg.emb_dim],
            initializer=KGPL_STUDENT.get_initializer(),
            name="entity_emb_matrix",
        )
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.cfg.emb_dim],
            initializer=KGPL_STUDENT.get_initializer(),
            name="relation_emb_matrix",
        )

        # [batch_size, dim]
        self.user_embeddings = tf.nn.dropout(
            tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices), 1 - self.dropout_rate
        )

        entities, relations = self.get_neighbors(self.item_indices)
        self.batch_entities = entities
        self.entity_indices = tf.concat([tf.reshape(es, [-1]) for es in entities], 0)
        self.entity_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.entity_indices)
        self.relation_indices = tf.concat([tf.reshape(rs, [-1]) for rs in relations], 0)
        self.relation_embeddings = tf.nn.embedding_lookup(
            self.relation_emb_matrix, self.relation_indices
        )

        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.cfg.n_iter):
            neighbor_entities = tf.reshape(
                tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1]
            )
            neighbor_relations = tf.reshape(
                tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1]
            )
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]
        for i in range(self.cfg.n_iter):
            if i == self.cfg.n_iter - 1:
                aggregator = SumAggregatorWithDropout(
                    self.batch_size,
                    self.cfg.emb_dim,
                    self.dropout_rate,
                    act=tf.nn.tanh,
                    cfg=self.cfg,
                    name=f"agg_{i}",
                )
            else:
                aggregator = SumAggregatorWithDropout(
                    self.batch_size,
                    self.cfg.emb_dim,
                    self.dropout_rate,
                    act=tf.nn.leaky_relu,
                    cfg=self.cfg,
                    name=f"agg_{i}",
                )
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.cfg.n_iter - i):
                shape = [self.batch_size, -1, self.cfg.neighbor_sample_size, self.cfg.emb_dim]
                vector = aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                    user_embeddings=self.user_embeddings,
                    masks=None,
                )
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.cfg.emb_dim])

        return res, aggregators

    def _build_train(self, scope):
        """Implementation of the risk and optimizer.
        """
        # compute losses for all samples and then split those into positve, negative, and pseudo-labelled by using binary masks
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores)
        loss_obs = loss * (1 - self.neg_mask) * (1 - self.plabel_mask)
        loss_uobs = loss * self.neg_mask
        loss_pl = loss * self.plabel_mask

        # compute the risk
        self.obs_loss = tf.reduce_mean(loss_obs)
        self.uobs_loss = tf.reduce_mean(loss_uobs)
        self.pl_loss = tf.reduce_mean(loss_pl)
        self.base_loss = tf.reduce_sum(loss_obs + loss_uobs + loss_pl)

        self.loss = self.base_loss
        self.opt = tf.train.AdamOptimizer(self.cfg.optimize.lr)

        # to check gradients
        tvs = tf.trainable_variables(self.name)
        self.accum_vars_obs = [
            (tv.name, tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False))
            for tv in tvs
        ]
        self.accum_vars_uobs = [
            (tv.name, tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False))
            for tv in tvs
        ]
        self.accum_vars_pl = [
            (tv.name, tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False))
            for tv in tvs
        ]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for _, tv in self.accum_vars_obs]
        self.zero_ops += [tv.assign(tf.zeros_like(tv)) for _, tv in self.accum_vars_uobs]
        self.zero_ops += [tv.assign(tf.zeros_like(tv)) for _, tv in self.accum_vars_pl]

        gvs_obs = self.opt.compute_gradients(self.obs_loss, tvs)
        gvs_uobs = self.opt.compute_gradients(self.uobs_loss, tvs)
        gvs_pl = self.opt.compute_gradients(self.pl_loss, tvs)

        self.accum_ops = [
            self.accum_vars_obs[i][1].assign_add(tf.math.abs(gv[0])) for i, gv in enumerate(gvs_obs)
        ]
        self.accum_ops += [
            self.accum_vars_uobs[i][1].assign_add(tf.math.abs(gv[0]))
            for i, gv in enumerate(gvs_uobs)
        ]
        self.accum_ops += [
            self.accum_vars_pl[i][1].assign_add(tf.math.abs(gv[0])) for i, gv in enumerate(gvs_pl)
        ]
        self.optimizer = self.opt.minimize(self.loss)

    def train(self, sess, feed_dict):
        _, loss, obs_loss, uobs_loss, pl_loss, _, grad_obs, grad_uobs, grad_pl = sess.run(
            [
                self.optimizer,
                self.base_loss,
                self.obs_loss,
                self.uobs_loss,
                self.pl_loss,
                self.accum_ops,
                [v for n, v in self.accum_vars_obs],
                [v for n, v in self.accum_vars_uobs],
                [v for n, v in self.accum_vars_pl],
            ],
            feed_dict,
        )
        grad_dict_obs = dict(zip([vname for vname, _ in self.accum_vars_obs], grad_obs))
        grad_dict_uobs = dict(zip([vname for vname, _ in self.accum_vars_uobs], grad_uobs))
        grad_dict_pl = dict(zip([vname for vname, _ in self.accum_vars_pl], grad_pl))
        user_grad_obs, entity_grad_obs = (
            grad_dict_obs[f"{self.name}/user_emb_matrix:0"],
            grad_dict_obs[f"{self.name}/entity_emb_matrix:0"],
        )
        user_grad_uobs, entity_grad_uobs = (
            grad_dict_uobs[f"{self.name}/user_emb_matrix:0"],
            grad_dict_uobs[f"{self.name}/entity_emb_matrix:0"],
        )
        user_grad_pl, entity_grad_pl = (
            grad_dict_pl[f"{self.name}/user_emb_matrix:0"],
            grad_dict_pl[f"{self.name}/entity_emb_matrix:0"],
        )
        return {
            "loss": loss,
            "obs_loss": obs_loss,
            "uobs_loss": uobs_loss,
            "pl_loss": pl_loss,
            "user_grad_obs": np.mean(np.sum(user_grad_obs, 1)),
            "entity_grad_obs": np.mean(np.sum(entity_grad_obs, 1)),
            "user_grad_uobs": np.mean(np.sum(user_grad_uobs, 1)),
            "entity_grad_uobs": np.mean(np.sum(entity_grad_uobs, 1)),
            "user_grad_pl": np.mean(np.sum(user_grad_pl, 1)),
            "entity_grad_pl": np.mean(np.sum(entity_grad_pl, 1)),
        }

    def get_feed_dict(self, data, epoch=0, eval_mode=False, sess=None):
        if not eval_mode:
            # Sample "1/3 batch-size" users 
            users = np.random.choice(
                self.all_users, self.cfg.optimize.batch_size // 3, replace=False
            )
            # Sample positive items
            items = [random.choice(tuple(self.user_seed_dict[u])) for u in users]

            # Sample pseudo-labelled items
            pl_users, pl_items, pl_labels = self._get_mini_batch_pl(sess, users)

            # Sample negative items
            seen_pair = set(zip(users, items)) | set(zip(pl_users, pl_items))
            cands, F = self.item_freq
            neg_items = []
            for u in users:
                while True:
                    j = cands[np.searchsorted(F, random.random())]
                    if j not in self.user_seed_dict[u] and j not in seen_pair:
                        break
                neg_items.append(j)

            # Create masks for positve, negative, pseudo-labelled instances in the mini-batch
            labels = [1] * len(items) + [0] * len(neg_items)
            plabel_mask = [0] * len(items) + [0] * len(neg_items) + [1] * len(pl_items)
            neg_mask = [0] * len(items) + [1] * len(neg_items) + [0] * len(pl_items)

            all_users = np.concatenate([users, users, pl_users])
            all_items = np.concatenate([items, neg_items, pl_items])

            feed_dict = {
                self.user_indices: all_users,
                self.item_indices: all_items,
                self.labels: np.concatenate([labels, pl_labels]),
                self.plabel_mask: plabel_mask,
                self.neg_mask: neg_mask,
                self.dropout_rate: self.cfg.dropout_rate,
            }
        else:
            feed_dict = {
                self.user_indices: data[:, 0],
                self.item_indices: data[:, 1],
                self.labels: data[:, 2],
                self.dropout_rate: 0.0,
            }
        return feed_dict


class KGPL_COT(object):
    def __init__(
        self,
        cfg,
        n_user, n_item, n_entity, n_relation,
        adj_entity, adj_relation, path_list_dict,
        train_data, eval_data, eval_mode=False
    ):
        self.cfg = cfg
        _cfg = copy.deepcopy(cfg)

        # Create two models
        self.f = KGPL_STUDENT(
            _cfg,
            n_user, n_entity, n_relation,
            adj_entity, adj_relation, path_list_dict,
            name="f", eval_mode=eval_mode,
        )
        self.g = KGPL_STUDENT(
            _cfg,
            n_user, n_entity, n_relation,
            adj_entity, adj_relation, path_list_dict,
            name="g", eval_mode=eval_mode,
        )
        if not eval_mode:
            _pos_inds = train_data[:, 2] == 1
            _pos_inds_ev = eval_data[:, 2] == 1
            self.f.set_item_candidates(
                n_user,
                n_item,
                train_data[_pos_inds],
                eval_data[_pos_inds_ev],
                path_list_dict,
            )

            # copy data from f to g
            self.g.n_item = self.f.n_item
            self.g.all_items = self.f.all_items
            self.g.all_users = self.f.all_users
            self.g.item_freq = self.f.item_freq
            self.g.neg_c_dict_user = self.f.neg_c_dict_user
            self.g.neg_c_dict_item = self.f.neg_c_dict_item
            self.g.user_seed_dict = self.f.user_seed_dict
            self.g.item_dist_dict = self.f.item_dist_dict

    def get_feed_dict(self, data, start, end, epoch=0, eval_mode=False, sess=None):
        if not eval_mode:
            feed_dict = {"data": data, "start": start, "end": end, "epoch": epoch}
        else:
            feed_dict = self.f.get_feed_dict(data[start:end], eval_mode=True, sess=sess)
        return feed_dict

    def split_data(self, data):
        N = len(data) // 2
        return data[:N], data[N:]

    def get_swap_feed_dict(self, sess, meta_feed_dict):
        fd = meta_feed_dict

        # Split meta feed-dict
        s, e = fd["start"], fd["end"]
        log_f, log_g = self.split_data(fd["data"][s:e])

        # Construct feed-dicts augmented with pseudo-labels
        f_fd = self.f.get_feed_dict(log_f, epoch=fd["epoch"], eval_mode=False, sess=sess)
        g_fd = self.g.get_feed_dict(log_g, epoch=fd["epoch"], eval_mode=False, sess=sess)

        # Target entries of a feed-dict
        fd_entries = [
            "user_indices",
            "item_indices",
            "labels",
            "plabel_mask",
            "neg_mask",
            "dropout_rate",
        ]

        # Exchange feed-dicts
        f_fd_train, g_fd_train = {}, {}
        for ky in fd_entries:
            f_fd_train[getattr(self.f, ky)] = g_fd[getattr(self.g, ky)]
            g_fd_train[getattr(self.g, ky)] = f_fd[getattr(self.f, ky)]
        return f_fd_train, g_fd_train

    def train(self, sess, meta_feed_dict):
        # Get augmented and swapped feed-dicts
        f_fd_train, g_fd_train = self.get_swap_feed_dict(sess, meta_feed_dict)

        # Train two models
        f_results = self.f.train(sess, f_fd_train)
        g_results = self.g.train(sess, g_fd_train)
        
        # Aggregate summaries
        ret_dict = {}
        for name, res in zip(["f_", "g_"], [f_results, g_results]):
            ret_dict.update({name + k: v for k, v in res.items()})
        return ret_dict

    def get_scores(self, sess, users, items, get_emb=False):
        # For evaluation, use only model f
        N = self.f.cfg.optimize.batch_size
        scores = sess.run(
            self.f.scores_normalized,
            {
                self.f.user_indices: users[:N],
                self.f.item_indices: items[:N],
                self.f.dropout_rate: 0.0,
            },
        )
        return items, scores

    def eval(self, sess, feed_dict):
        N = self.f.cfg.optimize.batch_size
        scores = sess.run(
            self.f.scores_normalized,
            {
                self.f.user_indices: feed_dict[self.f.user_indices][:N],
                self.f.item_indices: feed_dict[self.f.item_indices][:N],
                self.f.dropout_rate: 0.0,
            },
        )
        labels = feed_dict[self.f.labels]
        raw_scores = scores.copy()
        try:
            auc = roc_auc_score(y_true=labels, y_score=scores)
        except ValueError:
            auc = np.nan
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1, labels, raw_scores
