import tensorflow as tf

from model.triplet_loss import _pairwise_distances

N_CORRECT = 0
N_ITEMS_SEEN = 0

def reset_accuracy():
    global N_CORRECT, N_ITEMS_SEEN
    N_CORRECT = 0
    N_ITEMS_SEEN = 0

def update_accuracy(embeddings, labels):
    global N_CORRECT, N_ITEMS_SEEN
    anchor = embeddings[0]
    positive = None
    for e, y in zip(embeddings[1:], labels):
        if y == labels[0]: # zero label
            positive = e

    assert positive != None
    pos_distance = tf.norm(anchor - positive)

    is_successful = True
    for e in embeddings[1:]:
        if e == positive: continue
        neg_distance = tf.norm(anchor - e)
        if pos_distance < neg_distance:
            is_successful = False

    if is_successful:
        N_CORRECT += 1
    N_ITEMS_SEEN += 1

def calculate_accuracy():
    global N_CORRECT, N_ITEMS_SEEN
    return float(N_CORRECT) / N_ITEMS_SEEN


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
