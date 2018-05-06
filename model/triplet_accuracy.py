import tensorflow as tf

from model.triplet_loss import _pairwise_distances

def calculate_accuracy(embeddings, labels):
    pairwise_dist = _pairwise_distances(embeddings)
    distance_vector = tf.squeeze(tf.slice(pairwise_dist, [1, 0], [-1, 1]))
    labels_vector = tf.slice(labels, [1], [-1])
    pos_index = tf.argmin(labels_vector)
    pos_distance = tf.slice(distance_vector, [pos_index], [1])
    num_successes = tf.count_nonzero(tf.less_equal(pos_distance, distance_vector))
    batch_size_minus_one = tf.squeeze(tf.slice(tf.shape(labels), [0] , [1])) - 1
    batch_size_minus_one = tf.cast(batch_size_minus_one, dtype=tf.int64)
    return tf.cast(tf.equal(num_successes, batch_size_minus_one), dtype=tf.uint8)


def calculate_easier_accuracy(embeddings, labels):
    pairwise_dist = _pairwise_distances(embeddings)
    distance_vector = tf.squeeze(tf.slice(pairwise_dist, [1, 0], [-1, 1]))
    labels_vector = tf.slice(labels, [1], [-1])
    pos_index = tf.argmin(labels_vector)
    pos_distance = tf.slice(distance_vector, [pos_index], [1])
    num_successes = tf.count_nonzero(tf.less_equal(pos_distance, distance_vector)) - 1
    batch_size_minus_two = tf.squeeze(tf.slice(tf.shape(labels), [0] , [1])) - 2
    batch_size_minus_two = tf.cast(batch_size_minus_two, dtype=tf.int64)
    return tf.divide(num_successes, batch_size_minus_two)
