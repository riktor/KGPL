import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=""):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name, cfg=None):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + "_" + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim
        self.cfg = cfg

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        outputs = self._call(
            self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks
        )
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # dimension:
        # self_vectors: [batch_size, -1, dim] ([batch_size, -1] for LabelAggregator)
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim] ([batch_size, -1, n_neighbor] for LabelAggregator)
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        # masks (only for LabelAggregator): [batch_size, -1]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [-1, 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            # if (self.cfg is None) or (self.cfg.score_type == 'dot_product'):
            #     user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
            # elif self.cfg.score_type == 'cosine':
            #     norm = tf.sqrt(tf.reduce_sum(tf.square(user_embeddings), -1)) * tf.sqrt(tf.reduce_sum(tf.square(neighbor_relations), -1))
            #     user_relation_scores = self.cfg.sigma_scaler * (tf.reduce_sum(user_embeddings * neighbor_relations, axis=-1) / norm)
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
            scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1]
            scores_normalized = tf.expand_dims(scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregatorWithDropout(Aggregator):
    def __init__(self, batch_size, dim, dropout=0.0, act=tf.nn.leaky_relu, name=None, cfg=None):
        super(SumAggregatorWithDropout, self).__init__(batch_size, dim, dropout, act, name, cfg)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim],
                initializer=tf.contrib.layers.xavier_initializer(),
                name="weights",
            )
            self.bias = tf.get_variable(
                shape=[self.dim], initializer=tf.zeros_initializer(), name="bias"
            )

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(
            neighbor_vectors, neighbor_relations, user_embeddings
        )
        # [-1, dim]
        self_vectors = tf.nn.dropout(self_vectors, 1 - self.dropout)
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])
        output = self.act(output)

        return output

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [-1, 1, 1, self.dim])

            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
            scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1]
            scores_normalized = tf.expand_dims(scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(scores_normalized * neighbor_vectors, axis=2)
            neighbors_aggregated = tf.nn.dropout(neighbors_aggregated, 1 - self.dropout)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class LabelAggregator(Aggregator):
    def __init__(self, batch_size, dim, name=None, cfg=None):
        super(LabelAggregator, self).__init__(batch_size, dim, 0.0, None, name, cfg)

    def _call(self, self_labels, neighbor_labels, neighbor_relations, user_embeddings, masks):
        # [batch_size, 1, 1, dim]
        user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

        # [batch_size, -1, n_neighbor]
        # if (self.cfg is None) or (self.cfg.score_type == 'dot_product'):
        #     user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
        # elif (self.cfg.score_type == 'cosine'):
        #     norm = tf.sqrt(tf.reduce_sum(tf.square(user_embeddings), -1)) * tf.sqrt(tf.reduce_sum(tf.square(neighbor_relations), -1))
        #     user_relation_scores = self.cfg.sigma_scaler * (tf.reduce_sum(user_embeddings * neighbor_relations, axis=-1) / norm)

        user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
        user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

        # [batch_size, -1]
        neighbors_aggregated = tf.reduce_mean(
            user_relation_scores_normalized * neighbor_labels, axis=-1
        )
        output = (
            tf.cast(masks, tf.float32) * self_labels
            + tf.cast(tf.logical_not(masks), tf.float32) * neighbors_aggregated
        )

        return output
