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
        if y == labels[0]:
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
    # TODO: Need to implement
    pairwise_dist = _pairwise_distances(embeddings)
    distance_vector = tf.slice(pairwise_dist, 0, 1)
